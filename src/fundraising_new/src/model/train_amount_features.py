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
        "pct_bachelor_plus": ["pct_bachelor_plus","pct_bachelors_plus","pct_ba_plus"],
        "median_hh_income":  ["median_hh_income","median_income","B19013_001E"],
        "urban_q":           ["urban_q","metro_micro","cbsa_cat_num","urban_simple"],
        "share_dem":         ["share_dem","p_donate_dem","party_dem_share"],  # loose
        "share_gop":         ["share_gop","p_donate_rep","party_rep_share"],
    }
    if key in df_cols: return key
    for cand in aliases.get(key, []):
        if cand in df_cols: return cand
    # exact alphanumeric match
    norm = lambda s: re.sub(r"[^a-z0-9]+","", s.lower())
    for c in df_cols:
        if norm(c) == norm(key):
            return c
    return None

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
        s = pd.to_numeric(cbg[col], errors="coerce")
        cols[key] = s
        used.append(key)

    if not used:
        # nothing mapped yet — return an empty frame; caller will fallback to defaults
        return pd.DataFrame({"zip5": cbg["zip5"].values}).drop_duplicates(), []

    out = pd.DataFrame(cols)
    Z = out.groupby("zip5").mean(numeric_only=True).reset_index()

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
    ap.add_argument("--alphas", default="0.1,0.3,1,3,10", help="Comma-separated Ridge alphas")
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
    if not used_keys:
        # Fallback to a sensible default set commonly present in CBG features
        fallback_keys = [
            "pct_bachelor_plus", "median_hh_income", "urban_q", "pct_broadband",
            "poverty_rate", "owner_occ_rate", "rent_as_income_pct",
            "share_dem", "share_gop", "ntee_public_affairs_per_1k",
            "emp_share_manufacturing", "emp_share_healthcare_social",
        ]
        Z_fallback, used_fallback = align_feature_matrix(cbg, fallback_keys)
        if not used_fallback:
            raise SystemExit("No usable CBG features found to learn e_size. "
                            "Check cbg_features columns or extend alias map.")
        print(f"[info] using fallback features for e_size: {used_fallback}")
        Z, used_keys = Z_fallback, used_fallback

    D = fec.merge(Z, on="zip5", how="inner").dropna(subset=["avg_amount_zip"])

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
    y = D["avg_amount_zip"].astype(float).values

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

    # 6) Normalize to sum |w| = 1 and write e_size table
    e_size = coefs_raw.copy()
    s = e_size.abs().sum()
    if s > 0: e_size = e_size / s
    basis_e = (e_size.rename("e_size").reset_index()
               .rename(columns={"index":"feature_key"}))
    # ensure feature_key lowercased to match basis
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
