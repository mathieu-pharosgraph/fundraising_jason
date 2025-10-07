#!/usr/bin/env python3
import argparse, pandas as pd, json, numpy as np

def _to_nullable_int(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    try:
        # Newer pandas
        return s.astype("Int64")
    except (TypeError, ValueError):
        # Older pandas: keep float64 (has NaN) and match on both sides
        return s.astype("float64")

def pct_notnull(series: pd.Series) -> float:
    if series.dtype == "O":
        return float(series.notna().mean() * 100)
    return float(pd.to_numeric(series, errors="coerce").notna().mean() * 100)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)   # CSV from s5
    ap.add_argument("--s4", required=True)         # political_classification_enriched.parquet
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    en = pd.read_csv(args.enriched)
    s4 = pd.read_parquet(args.s4)

    # Align dtypes on join keys
    en["period"] = en["period"].astype(str)
    s4["period"] = s4["period"].astype(str)
    en["cluster_id"] = _to_nullable_int(en.get("cluster_id"))
    s4["cluster_id"] = _to_nullable_int(s4["cluster_id"])

    # Keep latest s4 per (cluster_id, period) if processed_at exists
    if "processed_at" in s4.columns:
        s4 = s4.sort_values("processed_at").drop_duplicates(["cluster_id","period"], keep="last")

    keep = [c for c in [
        "classification","dem_angle","gop_angle",
        "dem_fundraising_potential","gop_fundraising_potential"
    ] if c in s4.columns]

    merged = en.merge(
        s4[["period","cluster_id"] + keep],
        on=["period","cluster_id"], how="left", suffixes=("","_s4")
    )

    # Coerce numeric potentials
    for c in ["dem_fundraising_potential","gop_fundraising_potential","dem_fundraising_potential_s4","gop_fundraising_potential_s4"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Fill from _s4 columns only where base is null; report coverage
    print(f"✓ rows before fill: {len(merged)}")
    for c in ["classification","dem_angle","gop_angle",
              "dem_fundraising_potential","gop_fundraising_potential"]:
        c_s4 = f"{c}_s4"
        if c_s4 in merged.columns:
            before = pct_notnull(merged[c])
            src    = pct_notnull(merged[c_s4])
            merged[c] = merged[c].where(merged[c].notna(), merged[c_s4])
            after  = pct_notnull(merged[c])
            merged.drop(columns=[c_s4], inplace=True, errors="ignore")
            print(f"coverage {c}: {after:.1f}% (was {before:.1f}%, s4 source {src:.1f}%)")

    merged.to_csv(args.out, index=False)
    print(f"✓ merge_s4_into_enriched wrote {args.out} rows={len(merged)}")

if __name__ == "__main__":
    main()
