#!/usr/bin/env python3
"""
enrich_and_train_amount.py — one-shot helper to:
  1) enrich CBG features with missing ACS/NAICS columns,
  2) run train_amount_features.py (Ridge + RF blend),
  3) normalize/clean e_size in the basis,
  4) write a top blended e_size debug CSV (posthoc).

Usage (typical):
python fundraising_participation/src/model/enrich_and_train_amount.py \
  --cbg-features fundraising_participation/data/geo/cbg_features.parquet \
  --acs-viz "fundraising_participation/data/outputs_exp/viz_frame_zcta.parquet" \
  --acs-viz-fallback "fundraising_participation/data/outputs_exp/viz_frame_zcta.csv" \
  --naics-shares "fundraising_participation/data/geo/zcta_naics_shares.parquet" \
  --fec-zip fundraising_participation/data/fec/agg_zip/fec_zip_2020.csv \
  --basis-in data/donors/feature_basis.parquet \
  --basis-out data/donors/feature_basis.parquet \
  --outdir data/donors/amount_model \
  --alphas 0.03,0.1,0.3,1,3
"""

import argparse, subprocess, sys, pathlib, re
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- small helpers ----------
def _z5(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)

def enrich_cbg_with_acs(cbg_path: str, acs_parquet: str|None, acs_csv: str|None) -> list[str]:
    """Add owner_occ_rate and poverty_rate to cbg_features if available in ACS; derive if raw parts exist."""
    added = []
    cbg = pd.read_parquet(cbg_path)
    cbg["zcta"] = _z5(cbg["zcta"])
    acs = None
    if acs_parquet and Path(acs_parquet).exists():
        acs = pd.read_parquet(acs_parquet)
    elif acs_csv and Path(acs_csv).exists():
        acs = pd.read_csv(acs_csv, dtype=str, low_memory=False)
    if acs is None:
        return added

    acs = acs.copy()
    if "zcta" not in acs.columns:
        # attempt to detect any zcta-like col
        zcand = next((c for c in acs.columns if c.lower() in ("zcta","zip","zip5","zcta5","zipcode","zcta5ce10","geoid10","geoid")), None)
        if not zcand:
            return added
        acs["zcta"] = _z5(acs[zcand])
    acs["zcta"] = _z5(acs["zcta"])

    # direct if present
    for c in ("owner_occ_rate","poverty_rate"):
        if c in acs.columns:
            acs[c] = pd.to_numeric(acs[c], errors="coerce")

    # derive owner_occ_rate if needed
    if "owner_occ_rate" not in acs.columns:
        own_num = next((c for c in acs.columns if re.fullmatch(r"(?i)(B25003_002E|owner_occ_num|owner_occ_households)", c)), None)
        own_den = next((c for c in acs.columns if re.fullmatch(r"(?i)(B25003_001E|households_total)", c)), None)
        if own_num and own_den:
            num = pd.to_numeric(acs[own_num], errors="coerce")
            den = pd.to_numeric(acs[own_den], errors="coerce").replace(0, np.nan)
            acs["owner_occ_rate"] = (num/den * 100).clip(0,100)

    # derive poverty_rate if needed
    if "poverty_rate" not in acs.columns:
        pov_num = next((c for c in acs.columns if re.fullmatch(r"(?i)(poverty_num|below_poverty|B17001_pov_sum)", c)), None)
        pov_den = next((c for c in acs.columns if re.fullmatch(r"(?i)(pop_total|population|B17001_001E)", c)), None)
        if pov_num and pov_den:
            num = pd.to_numeric(acs[pov_num], errors="coerce")
            den = pd.to_numeric(acs[pov_den], errors="coerce").replace(0, np.nan)
            acs["poverty_rate"] = (num/den * 100).clip(0,100)

    keep = [c for c in ["zcta","owner_occ_rate","poverty_rate"] if c in acs.columns]
    if len(keep) > 1:
        add = acs[keep].copy()
        out = cbg.merge(add, on="zcta", how="left")
        out.to_parquet(cbg_path, index=False)
        if "owner_occ_rate" in keep: added.append("owner_occ_rate")
        if "poverty_rate" in keep:   added.append("poverty_rate")
    return added

def enrich_cbg_with_naics(cbg_path: str, naics_path: str|None) -> list[str]:
    """Merge NAICS employment shares (0..1) by ZCTA if provided."""
    added = []
    if not naics_path or not Path(naics_path).exists():
        return added
    cbg = pd.read_parquet(cbg_path)
    emp = pd.read_parquet(naics_path)
    cbg["zcta"] = _z5(cbg["zcta"])
    emp["zcta"] = _z5(emp["zcta"])
    want = ["zcta","emp_share_manufacturing","emp_share_retail","emp_share_healthcare_social",
            "emp_share_professional_scientific_mgmt","emp_share_information"]
    have = [c for c in want if c in emp.columns]
    if len(have) > 1:
        add = emp[have].copy()
        for c in have:
            if c != "zcta":
                add[c] = pd.to_numeric(add[c], errors="coerce").clip(0,1)
        out = cbg.merge(add, on="zcta", how="left")
        out.to_parquet(cbg_path, index=False)
        added += [c for c in have if c != "zcta"]
    return added

def run(cmd: list[str]) -> None:
    print("→", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        sys.exit(res.returncode)

def enrich_cbg_with_industry_shares_from_acs(cbg_path: str, acs_parquet: str|None, acs_csv: str|None) -> list[str]:
    """
    Derive industry employment shares per ZCTA from ACS 'viz_frame_zcta' (or similar) and merge into CBG.
    Will use shares directly if present (emp_share_*), else try to derive from counts.
    Returns list of columns added.
    """
    added = []
    cbg = pd.read_parquet(cbg_path)
    cbg["zcta"] = cbg["zcta"].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)

    # Load ACS
    acs = None
    if acs_parquet and Path(acs_parquet).exists():
        acs = pd.read_parquet(acs_parquet)
    elif acs_csv and Path(acs_csv).exists():
        acs = pd.read_csv(acs_csv, dtype=str, low_memory=False)
    if acs is None:
        return added

    acs = acs.copy()
    zcand = next((c for c in acs.columns if c.lower() in ("zcta","zip","zip5","zcta5","zipcode","zcta5ce10","geoid10","geoid")), None)
    if not zcand:
        return added
    acs["zcta"] = acs[zcand].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)

    # Preferred share columns if they already exist in ACS:
    SHARE_KEYS = [
        "emp_share_manufacturing",
        "emp_share_retail",
        "emp_share_healthcare_social",
        "emp_share_professional_scientific_mgmt",
        "emp_share_information",
        # optionally keep these if useful:
        "emp_share_agriculture",
        "emp_share_other_services",
    ]

    have_shares = [c for c in SHARE_KEYS if c in acs.columns]
    if have_shares:
        add = acs[["zcta"] + have_shares].copy()
        for c in have_shares:
            add[c] = pd.to_numeric(add[c], errors="coerce").clip(0, 1)  # keep 0..1
        out = cbg.merge(add, on="zcta", how="left")
        out.to_parquet(cbg_path, index=False)
        added += have_shares
        return added

    # Otherwise: attempt to derive shares from counts if present
    # Try to detect a 'total employment' column
    total_cands = [c for c in acs.columns if re.fullmatch(r"(?i)(emp_total|total_employed|employment_total)", c)]
    if not total_cands:
        # sometimes ACS frames include individual NAICS counts but no explicit total; skip derivation then
        return added
    emp_total = pd.to_numeric(acs[total_cands[0]], errors="coerce")

    # Candidate count columns (very permissive regex)
    def _num(col):
        return pd.to_numeric(acs[col], errors="coerce") if col in acs.columns else None

    cand_map = {
        "emp_share_manufacturing":               [c for c in acs.columns if re.search(r"(?i)emp.*manufactur", c)],
        "emp_share_retail":                      [c for c in acs.columns if re.search(r"(?i)emp.*retail", c)],
        "emp_share_healthcare_social":           [c for c in acs.columns if re.search(r"(?i)emp.*(health|social)", c)],
        "emp_share_professional_scientific_mgmt":[c for c in acs.columns if re.search(r"(?i)emp.*(prof|scient|mgmt|manage)", c)],
        "emp_share_information":                 [c for c in acs.columns if re.search(r"(?i)emp.*informat", c)],
        "emp_share_agriculture":                 [c for c in acs.columns if re.search(r"(?i)emp.*(agric|extract|mining)", c)],
        "emp_share_other_services":              [c for c in acs.columns if re.search(r"(?i)emp.*other.*service", c)],
    }

    deriv = {"zcta": acs["zcta"].values}
    for out_name, cands in cand_map.items():
        val = None
        for c in cands:
            s = _num(c)
            if s is not None:
                val = s if val is None else (val.fillna(0) + s.fillna(0))
        if val is not None:
            share = (val / emp_total.replace(0, np.nan)).clip(0, 1)
            deriv[out_name] = share

    add = pd.DataFrame(deriv)
    have = [c for c in add.columns if c != "zcta" and add[c].notna().any()]
    if have:
        out = cbg.merge(add[["zcta"] + have], on="zcta", how="left")
        out.to_parquet(cbg_path, index=False)
        added += have
    return added

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbg-features", required=True)
    ap.add_argument("--acs-viz", default=None, help="viz_frame_zcta parquet with ACS")
    ap.add_argument("--acs-viz-fallback", default=None, help="CSV fallback for viz_frame_zcta")
    ap.add_argument("--naics-shares", default=None, help="optional NAICS shares parquet by ZCTA")

    ap.add_argument("--fec-zip", required=True)
    ap.add_argument("--basis-in", required=True)
    ap.add_argument("--basis-out", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--alphas", default="0.03,0.1,0.3,1,3")

    # paths to your existing scripts
    ap.add_argument("--train-script", default="fundraising_participation/src/model/train_amount_features.py")
    ap.add_argument("--fix-script",   default="fundraising_participation/src/model/fix_basis_esize.py")
    args = ap.parse_args()

    # 1) enrich cbg_features
    added_acs   = enrich_cbg_with_acs(args.cbg_features, args.acs_viz, args.acs_viz_fallback)
    added_ind = enrich_cbg_with_industry_shares_from_acs(
        args.cbg_features, args.acs_viz, args.acs_viz_fallback
    )
    print("✓ industry shares from ACS:", (",".join(added_ind) if added_ind else "none"))

    added_naics = enrich_cbg_with_naics(args.cbg_features, args.naics_shares)
    print("✓ CBG enrichment:",
          ("ACS→" + ",".join(added_acs) if added_acs else "ACS none"),
          "|",
          ("NAICS→" + ",".join(added_naics) if added_naics else "NAICS none"))

    # 2) run trainer (Ridge+RF blend)
    run([
        sys.executable, args.train_script,
        "--fec-zip", args.fec_zip,
        "--cbg-features", args.cbg_features,
        "--basis-in", args.basis_in,
        "--outdir", args.outdir,
        "--basis-out", args.basis_out,
        "--alphas", args.alphas,
    ])

    # 3) coalesce e_size columns if needed
    run([sys.executable, args.fix_script, args.basis_out])

    # 4) posthoc blend debug (top features) — in case trainer didn’t write one
    try:
        B = pd.read_parquet(args.basis_out)
        top = (B[["feature_key","e_size"]]
               .assign(abs=lambda d: d["e_size"].abs())
               .sort_values("abs", ascending=False)
               .drop(columns=["abs"])
               .head(20))
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        top.to_csv(Path(args.outdir)/"amount_esize_blend_debug_posthoc.csv", index=False)
        print("✓ wrote", Path(args.outdir)/"amount_esize_blend_debug_posthoc.csv")
        print(top.to_string(index=False))
    except Exception as e:
        print("(!) could not write posthoc blend debug:", e)

if __name__ == "__main__":
    main()
