#!/usr/bin/env python3
"""
train_amount_features.py — learn feature weights for expected donation size (e_size)
using Ridge regression at the ZIP level, then merge into the feature basis.

Inputs
------
--fec-zip         CSV with ZIP-level totals or itemized amounts
                   Must contain either:
                     - itemized: columns ['zip5' (or zip), 'amount']  OR
                     - aggregated: ['zip5','total_amount','n_contribs']
--cbg-features    Parquet with CBG features (must contain 'cbg_id' and 'zcta')
--basis-in        Existing feature basis parquet (with p_any, p_dem_given, ...)
--outdir          Where to write model artifacts
--basis-out       Output path for updated feature basis parquet (default = basis-in)

What it does
------------
1) Builds ZIP5 average amounts from FEC input (robust to column names).
2) Aggregates CBG features to ZIP5 (mean).
3) Aligns features to the basis feature_key set (case/alias tolerant).
4) Fits RidgeCV on standardized features to predict avg_amount_zip.
5) Exports normalized signed coefficients as e_size (sum |w| = 1).
6) Merges e_size into the basis parquet.

Usage
-----
python src/model/train_amount_features.py \
  --fec-zip fundraising_participation/data/fec/zip_amounts_agg.csv \
  --cbg-features fundraising_participation/data/geo/cbg_features.parquet \
  --basis-in data/donors/feature_basis.parquet \
  --outdir data/donors/amount_model \
  --basis-out data/donors/feature_basis.parquet
"""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ---------- helpers ----------
def normalize_zip5(s):
    s = pd.Series(s, copy=False).astype(str).str.strip().str.replace(r"\.0$","",regex=True)
    return s.str.extract(r"(\d{5})")[0].astype("string")

def load_fec_zip_table(path):
    """
    Return df with zip5, avg_amount_zip
    Accepts either itemized 'amount' or aggregated 'total_amount' + 'n_contribs'.
    """
    df = pd.read_csv(path, low_memory=False)
    # find ZIP column
    zip_col = None
    for c in ["zip5","zip","ZIP","Zip","zcta5","ZCTA5"]:
        if c in df.columns:
            zip_col = c; break
    if zip_col is None:
        raise SystemExit("FEC CSV must include a ZIP-like column (zip/zip5/ZCTA5).")
    df["zip5"] = normalize_zip5(df[zip_col])

    if "total_amount" in df.columns and "n_contribs" in df.columns:
        # already aggregated; collapse duplicates
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
        df["n_contribs"]   = pd.to_numeric(df["n_contribs"], errors="coerce")
        agg = (df.groupby("zip5", dropna=False)
                 .agg(total_amount=("total_amount","sum"),
                      n_contribs=("n_contribs","sum"))
                 .reset_index())
        agg["avg_amount_zip"] = agg["total_amount"] / agg["n_contribs"].replace(0,np.nan)
    elif "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        agg = (df.groupby("zip5", dropna=False)
                 .agg(total_amount=("amount","sum"),
                      n_contribs=("amount","size"))
                 .reset_index())
        agg["avg_amount_zip"] = agg["total_amount"] / agg["n_contribs"].replace(0,np.nan)
    else:
        raise SystemExit("FEC CSV must have 'amount' (itemized) or 'total_amount' & 'n_contribs' (aggregated).")

    # clamp and fill
    med = float(np.nanmedian(agg["avg_amount_zip"]))
    hi  = float(np.nanquantile(agg["avg_amount_zip"].dropna(), 0.99)) if agg["avg_amount_zip"].notna().any() else med
    agg["avg_amount_zip"] = agg["avg_amount_zip"].clip(lower=5.0, upper=hi).fillna(med)
    return agg[["zip5","avg_amount_zip"]]

def best_alias(df_cols, key):
    """Return best column name in df for a basis feature_key; handles common aliases."""
    aliases = {
        # BASIS KEYS  → CBG column candidates
        "income": ["median_hh_income","median_income","B19013_001E"],
        "educ": ["pct_bachelors_plus","pct_bachelor_plus","pct_ba_plus"],
        "internet_home": ["pct_broadband","broadband_rate"],
        "home_owner": ["owner_occ_rate"],
        "urban_simple": ["urban_q","metro_micro","cbsa_cat_num"],

        # if you keep these in basis too, include them
        "poverty_rate": ["poverty_rate"],
        "median_gross_rent": ["median_gross_rent"],
        "median_home_value": ["median_home_value"],
        "rent_as_income_pct": ["rent_as_income_pct"],
        "ntee_public_affairs_per_1k": ["ntee_public_affairs_per_1k"],
        "ntee_total_per_1k": ["ntee_total_per_1k"],

        # party mix (optional if present in CBG)
        "share_dem": ["share_dem"],
        "share_gop": ["share_gop"],

        # industries
        "emp_share_manufacturing": ["emp_share_manufacturing"],
        "emp_share_retail": ["emp_share_retail"],
        "emp_share_healthcare_social": ["emp_share_healthcare_social"],
        "emp_share_professional_scientific_mgmt": ["emp_share_professional_scientific_mgmt"],
        "emp_share_information": ["emp_share_information"],
        # (optionally)
        "emp_share_agriculture": ["emp_share_agriculture"],
        "emp_share_other_services": ["emp_share_other_services"],
    }
    # exact match first
    if key in df_cols:
        return key
    # try alias list for this basis key
    for cand in aliases.get(key, []):
        if cand in df_cols:
            return cand
    # final fallback: exact alphanumeric match
    norm = lambda s: re.sub(r"[^a-z0-9]+","", s.lower())
    for c in df_cols:
        if norm(c) == norm(key):
            return c
    return None

def _clean_feature_series(key: str, s: pd.Series) -> pd.Series:
    """Sanitize units/sentinels/ranges before aggregation."""
    s = pd.to_numeric(s, errors="coerce")

    # % / share variables → clamp to 0..100
    pct_like = {"educ","internet_home","rent_as_income_pct",
                "pct_bachelors_plus","pct_bachelor_plus","pct_broadband","pct_snap"}
    if key in pct_like or key.startswith("pct_"):
        return s.clip(0, 100)

    # money-ish variables in dollars → clamp to reasonable range
    if key in {"income","median_hh_income","median_income","B19013_001E",
               "median_gross_rent","median_home_value"}:
        s = s.mask(s <= 0, np.nan)                    # drop non-positive sentinels (e.g., -666666666)
        if key in {"median_hh_income","median_income","income","B19013_001E"}:
            return s.clip(5_000, 300_000)
        if key == "median_gross_rent":
            return s.clip(200, 5_000)
        if key == "median_home_value":
            return s.clip(10_000, 5_000_000)
        if key.startswith("emp_share_"):
            return s.clip(0, 1)

    # per-1k counts → trim fat tails
    if key in {"ntee_public_affairs_per_1k","ntee_total_per_1k"}:
        return s.clip(0, 300)

    # ownership rate / home_owner (if present as 0..1) → project to 0..100
    if key in {"home_owner","owner_occ_rate"}:
        if s.quantile(0.95) <= 1.5:  # looks like 0..1
            s = s * 100.0
        return s.clip(0, 100)

    # default passthrough
    return s


def align_feature_matrix(cbg, basis_keys):
    """
    Build ZIP-level matrix for features in basis_keys.
    Aggregation = mean over CBGs in the ZIP (via zcta→zip5).
    Returns (Z, used_keys): Z has columns ['zip5', used_keys...]
    """
    if "zcta" not in cbg.columns:
        raise SystemExit("--cbg-features must include 'zcta'.")
    cbg["zip5"] = cbg["zcta"].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)

    used = []
    cols = {"zip5": cbg["zip5"]}
    for key in basis_keys:
        col = best_alias(cbg.columns, key)
        if col is None:
            continue
        s = _clean_feature_series(key, cbg[col])
        cols[key] = s
        used.append(key)

    if not used:
        # nothing mapped yet — return an empty frame; caller will fallback to defaults
        return pd.DataFrame({"zip5": cbg["zip5"].values}).drop_duplicates(), []

    out = pd.DataFrame(cols)
    num_keys = [k for k in cols.keys() if k not in ("zip5","__w")]
    if "adults_18plus" in cbg.columns:
        out["__w"] = pd.to_numeric(cbg["adults_18plus"], errors="coerce").fillna(0.0).values
        wsum = out.groupby("zip5")["__w"].sum().replace(0, np.nan)
        num = out[num_keys].apply(pd.to_numeric, errors="coerce").multiply(out["__w"], axis=0)
        Z = num.groupby(out["zip5"]).sum().div(wsum, axis=0).reset_index().rename(columns={"zip5":"zip5"})
    else:
        Z = out.groupby("zip5", as_index=False)[num_keys].mean(numeric_only=True)
    # keep zip5 at the front
    Z = Z if "zip5" in Z.columns else Z.rename(columns={"index":"zip5"})






    # drop columns that are all-NaN or constant
    good = []
    for k in used:
        if k in Z.columns:
            s = Z[k]
            if s.notna().any() and s.nunique(dropna=True) > 1:
                good.append(k)
    Z = Z[["zip5"] + good] if good else Z[["zip5"]]
    return Z, good


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fec-zip", required=True, help="FEC ZIP-level (or itemized) CSV")
    ap.add_argument("--cbg-features", required=True, help="CBG features parquet (must have cbg_id, zcta)")
    ap.add_argument("--basis-in", required=True, help="Existing feature basis parquet")
    ap.add_argument("--outdir", required=True, help="Directory for model artifacts")
    ap.add_argument("--basis-out", default=None, help="Output basis parquet (default = --basis-in)")
    ap.add_argument("--alphas", default="0.03,0.1,0.3,1,3", help="Comma-separated Ridge alphas")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    basis_out = Path(args.basis_out) if args.basis_out else Path(args.basis_in)

    # 1) Basis → keys to learn (use all feature_key present)
    B = pd.read_parquet(args.basis_in)
    basis_keys = B["feature_key"].dropna().astype(str).unique().tolist()

    # 2) Load data
    fec = load_fec_zip_table(args.fec_zip)  # zip5, avg_amount_zip
    cbg = pd.read_parquet(args.cbg_features)

    # 3) Build ZIP-level feature matrix aligned to basis
    Z, used_keys = align_feature_matrix(cbg, basis_keys)  # Z: zip5 + mapped features
    try:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        Z.to_parquet(Path(args.outdir) / "zip_matrix_debug.parquet", index=False)
        print(f"[debug] wrote {Path(args.outdir) / 'zip_matrix_debug.parquet'} with cols={list(Z.columns)[:12]}")
    except Exception as e:
        print(f"[debug] failed to write zip_matrix_debug: {e}")

    if not used_keys:
        # Fallback to a broader, interpretable set commonly present in CBG features
        fallback_keys = [
            # Socio-economic / demos
            "pct_bachelors_plus","median_hh_income","median_age","household_size_avg",
            "pct_snap","pct_broadband","poverty_rate","owner_occ_rate","median_gross_rent",
            "median_home_value","rent_as_income_pct",

            # Party mix (optional, if you keep them)
            "share_dem","share_gop",

            # Civic / nonprofit / platform
            "ntee_public_affairs_per_1k","ntee_total_per_1k",

            # Employment structure (examples — include the ones you have)
            "emp_share_manufacturing","emp_share_retail","emp_share_healthcare_social",
            "emp_share_professional_scientific_mgmt","emp_share_information",
        ]

        Z_fallback, used_fallback = align_feature_matrix(cbg, fallback_keys)
        if not used_fallback:
            raise SystemExit("No usable CBG features found to learn e_size. "
                            "Check cbg_features columns or extend alias map.")
        print(f"[info] using fallback features for e_size: {used_fallback}")
        Z, used_keys = Z_fallback, used_fallback

    D = fec.merge(Z, on="zip5", how="inner").dropna(subset=["avg_amount_zip"])

    aud = []
    for col in [c for c in D.columns if c not in ("zip5","avg_amount_zip")]:
        s = pd.to_numeric(D[col], errors="coerce")
        if s.notna().any():
            try:
                corr = float(np.corrcoef(s.fillna(0), D["avg_amount_zip"])[0,1])
            except Exception:
                corr = 0.0
            aud.append((col, float(s.std()), corr))
    if aud:
        A = (pd.DataFrame(aud, columns=["feature","std_zip","corr_with_amount"])
            .sort_values("std_zip", ascending=False))
        A.to_csv(Path(args.outdir)/"amount_feature_variance_corr.csv", index=False)
        print("[debug] wrote amount_feature_variance_corr.csv")



    # quick variance/correlation audit
    feat_cols = [col for col in D.columns if col not in ("zip5","avg_amount_zip")]

    # final feature set
    feat_cols = [c for c in D.columns if c not in ("zip5","avg_amount_zip")]
    # defensive prune: remove any remaining all-NaN/constant/non-numeric columns
    keep = []
    for c in feat_cols:
        s = pd.to_numeric(D[c], errors="coerce")
        if s.notna().any() and s.nunique(dropna=True) > 1:
            keep.append(c)
    feat_cols = keep
    if not feat_cols:
        raise SystemExit("After pruning, no valid numeric features remain for Ridge. "
                        "Confirm cbg_features and chosen keys.")

    X = D[feat_cols].copy()
    y = np.log1p(D["avg_amount_zip"].astype(float).values)  # instead of raw amount

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), feat_cols)
    ])
    alphas = [float(a) for a in str(args.alphas).split(",")]
    ridge = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    pipe = Pipeline([("pre", pre), ("lm", ridge)])
    pipe.fit(X, y)

    # 5) Recover coefficients back on standardized columns
    lm = pipe.named_steps["lm"]
    # Pre.transform(X) = (X - med)/std ; coef on standardized cols:
    # We can get them by fitting RidgeCV; scikit doesn’t expose back-transform directly from pipeline
    # But we can compute effective coefficients via: beta_raw = beta_std / std
    imp = pipe.named_steps["pre"].named_transformers_["num"].named_steps["imp"]
    sc  = pipe.named_steps["pre"].named_transformers_["num"].named_steps["sc"]

    X_imp = pd.DataFrame(imp.transform(X), columns=feat_cols, index=X.index)
    scale = pd.Series(sc.scale_, index=feat_cols).replace(0, 1.0)
    coefs_std = pd.Series(lm.coef_, index=feat_cols)
    coefs_raw = (coefs_std / scale).replace([np.inf, -np.inf], 0).fillna(0.0)

    # ---- RandomForest + permutation importances to widen e_size coverage (optional) ----
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    rf = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=3, n_jobs=-1, random_state=42
    )
    # Use the standardized imput/scale X the pipeline sees:
    X_std = pipe.named_steps["pre"].transform(X)
    rf.fit(X_std, y)

    # permutation importance on standardized X (robust to scale)
    pi = permutation_importance(rf, X_std, y, n_repeats=5, random_state=42, n_jobs=-1)
    rf_imp = pd.Series(pi.importances_mean, index=feat_cols).clip(lower=0.0)
    if rf_imp.sum() > 0:
        rf_imp = rf_imp / rf_imp.sum()

    # ridge weights on raw scale (coefs_raw) -> convert to positive importances
    ridge_imp = coefs_raw.abs()
    if ridge_imp.sum() > 0:
        ridge_imp = ridge_imp / ridge_imp.sum()

    # blend (50% ridge signal + 50% RF importance)
    blend = 0.5 * ridge_imp + 0.5 * rf_imp.reindex(ridge_imp.index).fillna(0.0)

    # restore signs from ridge for features where ridge sign was nonzero; otherwise keep positive
    sign = np.sign(coefs_raw).replace(0, 1.0)
    e_size_blend = (blend * sign)

    # ---- Use the RF + Ridge BLEND as final e_size ----
    e_size = e_size_blend.copy()
    # L1 normalize for comparability across targets
    s = e_size.abs().sum()
    if s > 0:
        e_size = e_size / s

    # DEBUG: write top blended weights
    (
        e_size.rename("e_size_blend")
            .reset_index().rename(columns={"index":"feature_key"})
            .sort_values("e_size_blend", key=abs, ascending=False)
            .head(20)
    ).to_csv(Path(args.outdir)/"amount_esize_blend_debug.csv", index=False)
    print("[debug] wrote", Path(args.outdir)/"amount_esize_blend_debug.csv")

    basis_e = (
        e_size.rename("e_size").reset_index().rename(columns={"index":"feature_key"})
    )
    basis_e["feature_key"] = basis_e["feature_key"].astype(str).str.lower()


    
    # --- Map CBG feature names to existing basis keys before merge ---
    alias_to_basis = {
        "pct_bachelors_plus": "educ",
        "pct_bachelor_plus":  "educ",
        "median_hh_income":   "income",
        "pct_broadband":      "internet_home",
        "urban_q":            "urban_simple",            # only if you include urban_q in training
        "owner_occ_rate":     "home_owner",
        # leave true geo-only drivers unmapped so we can append them (see below), e.g.:
        # "ntee_public_affairs_per_1k": (append as new basis key)
    }

    basis_e_mapped = basis_e.copy()
    basis_e_mapped["feature_key"] = basis_e_mapped["feature_key"].replace(alias_to_basis)
    # collapse duplicates created by mapping and re-normalize
    basis_e_mapped = (basis_e_mapped.groupby("feature_key", as_index=False)["e_size"].sum())
    s2 = basis_e_mapped["e_size"].abs().sum()
    if s2 > 0:
        basis_e_mapped["e_size"] = basis_e_mapped["e_size"] / s2

    coef_tbl = (coefs_raw.rename("coef").reset_index().rename(columns={"index":"feature_key"})
                .assign(abs=lambda d: d["coef"].abs())
                .sort_values("abs", ascending=False))
    coef_tbl.head(20).to_csv(Path(args.outdir)/"amount_coef_debug.csv", index=False)

    print("Top raw coefs (abs):")
    print(coef_tbl.head(15).to_string(index=False))

    # Save model artifacts
    pd.DataFrame({
        "alpha_chosen":[lm.alpha_],
        "zip_rows":[len(D)],
        "features_used":[len(feat_cols)],
        "r2_in_sample":[pipe.score(X, y)]
    }).to_csv(outdir/"amount_ridgecv_summary.csv", index=False)
    basis_e.to_parquet(outdir/"feature_basis_e_size.parquet", index=False)

    # 7) Merge into basis
    M = B.merge(basis_e_mapped, on="feature_key", how="left").fillna({"e_size": 0.0})

    # (optional) append geo-only features not in basis so they carry e_size
    present = set(M["feature_key"].astype(str))
    new_rows = []
    for _, r in basis_e.iterrows():
        k = str(r["feature_key"])
        if (k not in present) and (k not in alias_to_basis):  # not mapped and not present
            new_rows.append({
                "feature_key": k,
                "p_any": 0.0, "p_dem_given": 0.0, "p_gop_given": 0.0,
                "p_type_candidate": 0.0, "p_type_org": 0.0,
                "feature_pretty": k.replace("_"," ").title(),
                "transform": "zscore", "clip_lo": -3.0, "clip_hi": 3.0,
                "e_size": float(r["e_size"])
            })
    if new_rows:
        M = pd.concat([M, pd.DataFrame(new_rows)], ignore_index=True)

    M.to_parquet(basis_out, index=False)
    print(f"✓ wrote {basis_out} with e_size; rows={len(M)} | alpha={lm.alpha_:.3g}")


if __name__ == "__main__":
    main()
