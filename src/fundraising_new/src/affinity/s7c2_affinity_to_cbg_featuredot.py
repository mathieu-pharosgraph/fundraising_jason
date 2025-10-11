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
import pyarrow as pa
import pyarrow.parquet as pq

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

def score_name(colname: str) -> str:
    if colname == "w_dem_comb":  return "affinity_cbg_dem"
    if colname == "w_gop_comb":  return "affinity_cbg_gop"
    if colname == "w_cand_comb": return "affinity_cbg_cand"
    if colname == "w_org_comb":  return "affinity_cbg_org"
    if colname == "e_size":      return "affinity_cbg_size"
    if colname == "p_any":       return "affinity_cbg_any"
    # fallback (shouldn’t trigger in our set)
    return f"affinity_cbg_{colname}"

def _norm_key(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+","", str(s).lower())

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
    # --- HARD FILTER: keep only labels present in S4 political file ---
    if Path(args.s4).exists():
        S4 = pd.read_parquet(args.s4)
        lab_candidates = ["story_label","label","winner_label","best_label"]
        lab_s4 = next((c for c in lab_candidates if c in S4.columns), None)
        if lab_s4 is not None:
            # remove explicitly non-political
            S4 = S4[~S4["classification"].fillna("non-political").str.contains("non-political", case=False)]
            S["label_key"] = S["label"].astype(str).str.lower().str.replace(r"[^a-z0-9]+","", regex=True)
            S4["label_key"] = S4[lab_s4].astype(str).str.lower().str.replace(r"[^a-z0-9]+","", regex=True)
            keep = set(S4["label_key"].dropna().unique().tolist())
            before = len(S)
            S = S[S["label_key"].isin(keep)].drop(columns=["label_key"])
            print(f"political filter: {len(S)}/{before} rows kept in story_features")

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

    # ----- (Optional) prepare S4 map for per-batch blending -----
    s4_map = {}
    if args.blend_party_with_s4 and Path(args.s4).exists():
        S4 = pd.read_parquet(args.s4)
        label_candidates = ["label","story_label","winner_label","best_label"]
        lab_s4 = next((c for c in label_candidates if c in S4.columns), None)
        if lab_s4 is not None:
            # normalize period to ISO date string
            def _norm_period(x):
                d = pd.to_datetime(x, errors="coerce", utc=False)
                return d.dt.date.astype("string")
            S4b = S4.copy()
            if "period" in S4b.columns:
                S4b["period"] = _norm_period(S4b["period"])
            else:
                S4b["period"] = pd.Series(dtype="string")
            S4b["label_key"] = S4b[lab_s4].astype(str).apply(norm_key)
            for c in ["dem_fundraising_potential","gop_fundraising_potential"]:
                if c in S4b.columns:
                    S4b[c] = pd.to_numeric(S4b[c], errors="coerce")/100.0
            # last write wins; build a dict keyed by (period,label_key)
            for _, r in S4b.iterrows():
                p = str(r.get("period") or "")
                lk = str(r.get("label_key") or "")
                dem = float(r.get("dem_fundraising_potential") or 1.0)
                gop = float(r.get("gop_fundraising_potential") or 1.0)
                s4_map[(p, lk)] = (dem, gop)

    # ----- Prepare a ParquetWriter with a fixed schema -----
    score_names = [score_name(c) for c in want_cols]

    fields = [
        pa.field("period", pa.string()),
        pa.field("label",  pa.string()),
        pa.field("label_key", pa.string()),    # NEW
        pa.field("cbg_id", pa.string()),
    ] + [pa.field(n, pa.float32()) for n in score_names]


    # add final blended party columns if we might blend
    blend_party = bool(s4_map)
    if blend_party:
        if "affinity_cbg_dem" in score_names:
            fields.append(pa.field("final_affinity_cbg_dem", pa.float32()))
        if "affinity_cbg_gop" in score_names:
            fields.append(pa.field("final_affinity_cbg_gop", pa.float32()))

    schema = pa.schema(fields)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, schema)

    # ----- Stream one (period,label) at a time -----
    n_batches = 0
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
            "label_key": _norm_key(label),   # NEW
            "cbg_id": Z["cbg_id"],
        })


        # compute every requested track
        for colname in want_cols:
            basis_w = np.array(
                [float(BW.loc[k, colname]) if k in BW.index else 0.0 for k in keys],
                dtype=float,
            )
            dots = (M * (story_w * basis_w)).sum(axis=1)
            lift = np.clip(dots, -args.clip, args.clip)
            score = 1.0 * (1.0 + args.lam * lift)

            out_name = score_name(colname)
            base[out_name] = score.astype(np.float32)

        # optional S4 blend per batch
        if blend_party:
            p_iso = pd.to_datetime(base["period"], errors="coerce").dt.date.astype("string")
            lk = str(norm_key(label))
            dem,gop = s4_map.get((p_iso.iloc[0], lk), (1.0, 1.0))
            if "affinity_cbg_dem" in base.columns:
                base["final_affinity_cbg_dem"] = (base["affinity_cbg_dem"].astype(float) * float(dem)).astype(np.float32)
            if "affinity_cbg_gop" in base.columns:
                base["final_affinity_cbg_gop"] = (base["affinity_cbg_gop"].astype(float) * float(gop)).astype(np.float32)

        # ensure all schema columns present
        for c in schema.names:
            if c not in base.columns:
                base[c] = np.nan

        # order columns and write
        base = base[schema.names]
        tbl = pa.Table.from_pandas(base, schema=schema, preserve_index=False)
        writer.write_table(tbl)
        n_batches += 1

    writer.close()
    print(f"✓ wrote {args.out} (batches={n_batches}, rows≈large)")

if __name__ == "__main__":
    main()
