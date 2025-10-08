#!/usr/bin/env python3
"""
s5c_fetch_acs_industry_zcta.py
Fetch ACS 5-year (table C24030) industry employment counts for all ZCTAs,
compute shares (0..1), and write a parquet for downstream merges.

Requires: a Census API key in env CENSUS_API_KEY (recommended), but will also
work without for small pulls (subject to throttling).

Example:
  python s5c_fetch_acs_industry_zcta.py \
    --year 2022 \
    --out fundraising_participation/data/geo/zcta_naics_shares.parquet
"""
import argparse, os, time, requests as rq, pandas as pd
from pathlib import Path

# C24030 codes we’ll use (see ACS docs):
# 001E = total, 002E=Ag/Forestry/Fishing/Mining, 003E=Construction, 004E=Manufacturing,
# 005E=Wholesale, 006E=Retail, 007E=Transportation+Utilities, 008E=Information,
# 009E=Finance+RealEstate, 010E=Professional+Scientific+Mgmt+Admin+Waste,
# 011E=Educational+HealthCare+Social, 012E=Arts+Recreation+Accommodation+Food,
# 013E=Other services, 014E=Public Administration
VARS = {
    "C24030_001E": "emp_total",
    "C24030_002E": "emp_agriculture",   # we’ll call this "agriculture"
    "C24030_004E": "emp_manufacturing",
    "C24030_006E": "emp_retail",
    "C24030_008E": "emp_information",
    "C24030_010E": "emp_prof_sci_mgmt",
    "C24030_011E": "emp_healthcare_social",
    # optionally keep others if you want them later:
    # "C24030_003E": "emp_construction",
    # "C24030_005E": "emp_wholesale",
    # "C24030_007E": "emp_transport_utils",
    # "C24030_009E": "emp_finance_real",
    # "C24030_012E": "emp_arts_food",
    # "C24030_013E": "emp_other_services",
    # "C24030_014E": "emp_public_admin",
}

def fetch_acs_c24030(year: int, key: str | None):
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    get_vars = ",".join(["NAME"] + list(VARS.keys()))
    params = {
        "get": get_vars,
        "for": "zip code tabulation area:*"
    }
    if key:
        params["key"] = key
    r = rq.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    cols = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=cols)
    # normalize zcta
    df["zcta"] = df["zip code tabulation area"].astype(str).str.zfill(5)
    # numeric coercion
    for v in VARS.keys():
        df[v] = pd.to_numeric(df[v], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)  # pick a recent ACS 5-yr
    ap.add_argument("--out",  default="fundraising_participation/data/geo/zcta_naics_shares.parquet")
    args = ap.parse_args()

    key = os.getenv("CENSUS_API_KEY")
    try:
        df = fetch_acs_c24030(args.year, key)
    except Exception as e:
        # try one retry in case of transient throttling
        time.sleep(2.0)
        df = fetch_acs_c24030(args.year, key)

    # compute shares
    out = pd.DataFrame({"zcta": df["zcta"]})
    total = pd.to_numeric(df["C24030_001E"], errors="coerce").replace(0, pd.NA)

    def share(code: str):
        s = pd.to_numeric(df[code], errors="coerce")
        return (s / total).clip(0, 1)

    out["emp_share_agriculture"]                  = share("C24030_002E")
    out["emp_share_manufacturing"]                = share("C24030_004E")
    out["emp_share_retail"]                       = share("C24030_006E")
    out["emp_share_information"]                  = share("C24030_008E")
    out["emp_share_professional_scientific_mgmt"] = share("C24030_010E")
    out["emp_share_healthcare_social"]            = share("C24030_011E")

    # tidy & write
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"✓ wrote {args.out} — rows={len(out)}, cols={len(out.columns)}")

if __name__ == "__main__":
    main()
