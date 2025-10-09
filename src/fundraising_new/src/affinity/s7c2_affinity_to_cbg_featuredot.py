#!/usr/bin/env python3
"""
s7c2_affinity_to_cbg_featuredot.py — story→CBG scoring via feature dot-products

Inputs
------
--story-features  data/topics/story_feature_affinity.parquet
                  Columns: period, label, feature_key, weight (−1..+1), optional reason
--cbg-features    fundraising_participation/data/geo/cbg_features.parquet
--basis           data/donors/feature_basis.parquet
--out             data/affinity/topic_affinity_by_cbg_featuredot.parquet

What it computes (per period, label, cbg_id)
--------------------------------------------
affinity_cbg_dem   using w_dem_comb
affinity_cbg_gop   using w_gop_comb
affinity_cbg_cand  using w_cand_comb
affinity_cbg_org   using w_org_comb
affinity_cbg_size  using e_size

All use:   dot = Σ_k  story_w(k) * basis_w(k) * Z_cbg_zscore(k)
           score = base * (1 + λ * clip(dot, −L, +L))   with base=1.0 by default.

You can later multiply party tracks by your base Dem/GOP potentials if desired.
"""
import argparse, re, numpy as np, pandas as pd
from pathlib import Path

# feature keys we never project numerically at CBG level
SKIP_KEYS = {"state"}  # add more if needed, e.g., {"state", "region"}

def norm_key(s): return re.sub(r"[^a-z0-9]+","", str(s).lower())

def best_alias(cols, key):
    # never project certain keys
    if key == "state":
        return None
    aliases = {
        "educ":["pct_bachelors_plus","pct_bachelor_plus"],
        "income":["median_hh_income","median_income","B19013_001E"],
        "internet_home":["pct_broadband","broadband_rate"],
        "urban_simple":["urban_q","metro_micro","cbsa_cat_num"],
        "home_owner":["owner_occ_rate"],
    }
    if key in cols: 
        return key
    for cand in aliases.get(key, []):
        if cand in cols: 
            return cand
    nk = norm_key(key)
    for c in cols:
        if norm_key(c) == nk: 
            return c
    return None


def zscore(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    m = s.mean(skipna=True)
    sd = s.std(ddof=1, skipna=True)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    out = (s - m) / sd
    return out.clip(-3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story-features", required=True)
    ap.add_argument("--cbg-features",   required=True)
    ap.add_argument("--basis",          required=True)
    ap.add_argument("--out",            required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.25)
    ap.add_argument("--clip",   dest="clip", type=float, default=0.5)
    ap.add_argument("--s4", default="data/topics/political_classification_enriched.parquet",
                help="Parquet with period,label,dem/gop_fundraising_potential (0..100)")
    ap.add_argument("--blend-party-with-s4", action="store_true",
                    help="If set, multiply affinity_cbg_dem/gop by normalized S4 base potentials")
    args = ap.parse_args()

    # 1) Load inputs
    S = pd.read_parquet(args.story_features)   # period,label,feature_key,weight
    B = pd.read_parquet(args.basis)            # feature_key + weight columns
    C = pd.read_parquet(args.cbg_features)
    C["cbg_id"] = C["cbg_id"].astype(str)

    # normalize types & drop non-projectables (e.g., 'state')
    if "feature_key" in S.columns:
        S["feature_key"] = S["feature_key"].astype(str)
        S = S[~S["feature_key"].isin(SKIP_KEYS)]
    if "period" in S.columns:
        S["period"] = S["period"].astype(str)
    if "label" in S.columns:
        S["label"] = S["label"].astype(str)

    # 2) Which weight columns exist in basis
    want_cols = [c for c in ["w_dem_comb","w_gop_comb","w_cand_comb","w_org_comb","e_size"] if c in B.columns]

    if "p_any" in B.columns:
        want_cols.append("p_any")  # will be emitted as affinity_cbg_any
    want_cols = list(dict.fromkeys(want_cols))
    if not want_cols:
        raise SystemExit("Basis has no combined weights; add w_dem_comb/w_gop_comb or e_size first.")


    # 3) Prepare a CBG feature matrix for the union of feature_keys appearing in S∩B
    feat_keys = sorted(set(B["feature_key"].astype(str)) & set(S["feature_key"].astype(str)))

    colmap = {}
    for k in feat_keys:
        if k in SKIP_KEYS:
            continue
        col = best_alias(C.columns, k)
        if not col:
            continue
        s = pd.to_numeric(C[col], errors="coerce")
        # ↓ add this check
        if s.notna().sum() < 2:
            continue
        colmap[k] = col




    Z = C[["cbg_id"]].copy()
    for k, col in colmap.items():
        Z[k] = zscore(C[col])
    # 4) Build basis weight table keyed by feature_key
    BW = B.set_index("feature_key")[want_cols].fillna(0.0)

    # 5) Compute per (period,label) dot-products → scores per cbg
    out_rows = []
    for (period, label), g in S.groupby(["period","label"]):
        # sparse story weights on the intersecting keys
        sw = g.set_index("feature_key")["weight"].to_dict()
        keys = [k for k in colmap.keys() if k in sw]
        if not keys:
            continue
        # vectorize
        story_w = np.array([float(sw[k]) for k in keys], dtype=float)
        # matrix of CBG z-features for those keys
        M = Z[keys].to_numpy(dtype=float)   # (N_cbgs, K)

        for colname in want_cols:
            basis_w = np.array([float(BW.loc[k, colname]) if k in BW.index else 0.0 for k in keys], dtype=float)
            # elementwise product per feature → sum across K
            dots = (M * (story_w * basis_w)).sum(axis=1)  # (N_cbgs,)
            lift = np.clip(dots, -args.clip, args.clip)
            score = 1.0 * (1.0 + args.lam * lift)         # base=1.0; can be replaced per-party later
            out_name = ("affinity_cbg_" + colname.split("_",1)[1]) if colname.startswith("w_") else \
                    ("affinity_cbg_any" if colname=="p_any" else f"affinity_cbg_{colname}")
            out_rows.append(pd.DataFrame({
                "period": period, "label": label, "cbg_id": Z["cbg_id"],
                out_name: score.astype(np.float32),
            }))

    if not out_rows:
        pd.DataFrame(columns=["period","label","cbg_id"]).to_parquet(args.out, index=False)
        print(f"✓ wrote {args.out} (empty; no overlapping features)")
        return

    # build one wide frame per (period,label), then concat vertically
    wide_rows = []
    for (period, label), g in S.groupby(["period","label"]):
        sw = g.set_index("feature_key")["weight"].to_dict()
        keys = [k for k in colmap.keys() if k in sw]
        if not keys:
            continue

        story_w = np.array([float(sw[k]) for k in keys], dtype=float)
        M = Z[keys].to_numpy(dtype=float)  # (N_cbgs, K)

        base = pd.DataFrame({
            "period": str(period),
            "label":  str(label),
            "cbg_id": Z["cbg_id"],
        })

        for colname in want_cols:
            basis_w = np.array(
                [float(BW.loc[k, colname]) if k in BW.index else 0.0 for k in keys],
                dtype=float,
            )
            dots = (M * (story_w * basis_w)).sum(axis=1)
            lift = np.clip(dots, -args.clip, args.clip)
            score = 1.0 * (1.0 + args.lam * lift)
            out_name = (
                "affinity_cbg_" + colname.split("_", 1)[1] if colname.startswith("w_")
                else ("affinity_cbg_any" if colname == "p_any" else f"affinity_cbg_{colname}")
            )
            base[out_name] = score.astype(np.float32)

        wide_rows.append(base)

    if not wide_rows:
        pd.DataFrame(columns=["period","label","cbg_id"]).to_parquet(args.out, index=False)
        print(f"✓ wrote {args.out} (empty; no overlapping features)")
        return

    OUT = pd.concat(wide_rows, ignore_index=True)



    # -------- Optional party blending with S4 base potentials --------
    if args.blend_party_with_s4:
        s4p = Path(args.s4)
        if s4p.exists():
            S4 = pd.read_parquet(s4p)

            # choose any label-like column present
            label_candidates = ["label","story_label","winner_label","best_label"]
            lab_s4 = next((c for c in label_candidates if c in S4.columns), None)
            if lab_s4 is None:
                print("[warn] S4 has no label-like column. Skipping blend.")
            else:
                # normalize period to ISO (YYYY-MM-DD) on both sides
                def _norm_period(x):
                    d = pd.to_datetime(x, errors="coerce", utc=False, infer_datetime_format=True)
                    return d.dt.date.astype("string")

                OUT["period"] = _norm_period(OUT["period"])
                S4b = S4.copy()
                S4b["period"] = _norm_period(S4b["period"]) if "period" in S4b.columns else pd.Series(dtype="string")

                # normalize label keys on both sides
                OUT["label_key"] = OUT["label"].astype(str).apply(norm_key)
                S4b["label_key"] = S4b[lab_s4].astype(str).apply(norm_key)

                # scale bases to 0..1
                keep = ["period","label_key"]
                for c in ["dem_fundraising_potential","gop_fundraising_potential"]:
                    if c in S4b.columns:
                        S4b[c] = pd.to_numeric(S4b[c], errors="coerce")/100.0
                        keep.append(c)

                S4b = S4b[keep].drop_duplicates(["period","label_key"])
                OUT = OUT.merge(S4b, on=["period","label_key"], how="left")

                # safe defaults = 1.0 → no reweight
                base_dem = OUT.get("dem_fundraising_potential", pd.Series(1.0, index=OUT.index)).fillna(1.0)
                base_gop = OUT.get("gop_fundraising_potential", pd.Series(1.0, index=OUT.index)).fillna(1.0)
                if "affinity_cbg_dem" in OUT.columns:
                    OUT["final_affinity_cbg_dem"] = (OUT["affinity_cbg_dem"].astype(float) * base_dem).astype(np.float32)
                if "affinity_cbg_gop" in OUT.columns:
                    OUT["final_affinity_cbg_gop"] = (OUT["affinity_cbg_gop"].astype(float) * base_gop).astype(np.float32)

                OUT.drop(columns=["label_key"], inplace=True, errors="ignore")
        else:
            print(f"[warn] --blend-party-with-s4 set but {s4p} not found; skipping blend.")




    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    OUT.to_parquet(args.out, index=False)
    print(f"✓ wrote {args.out} rows={len(OUT)} cols={len(OUT.columns)}")

if __name__ == "__main__":
    main()
