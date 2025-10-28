#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------
OUT = Path("outputs/mrp/microcells_pums.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Inputs (expect 2020 5y PUMS, unzipped):
# data/pums/psam_pusa.csv, psam_pusb.csv, psam_husa.csv, psam_husb.csv
# If only A files exist, it will still work.
# -------------------------------------------------------------------
PUMS_DIR = Path("data/pums")

# Columns we actually need (keeps memory reasonable)
PERSON_COLS = ["SERIALNO","ST","AGEP","SEX","RAC1P","HISP","SCHL","PINCP","MAR"]
HOUSE_COLS  = ["SERIALNO","TEN"]

# -------------------------------------------------------------------
# NaN-safe recoders
# -------------------------------------------------------------------
def recode_age(x):
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v): return np.nan
    v = int(v)
    if v < 30: return "18-29"
    if v < 45: return "30-44"
    if v < 65: return "45-64"
    return "65+"

def recode_edu(schl):
    v = pd.to_numeric(schl, errors="coerce")
    if pd.isna(v): return np.nan
    v = int(v)
    # ACS PUMS SCHL 1..24
    if v <= 15: return "HS_or_less"   # <= HS
    if v <= 20: return "SomeCollege"  # some college / assoc
    return "BAplus"                   # BA+

def recode_income(pincp):
    v = pd.to_numeric(pincp, errors="coerce")
    if pd.isna(v): return "mid"
    v = float(v)
    if v < 35000: return "low"
    if v < 100000: return "mid"
    return "high"

def recode_race_eth(rac1p, hisp):
    h = pd.to_numeric(hisp, errors="coerce")
    # ACS PUMS HISP: 1 = Not Hispanic; else Hispanic
    if not pd.isna(h) and int(h) != 1:
        return "Latino"
    r = pd.to_numeric(rac1p, errors="coerce")
    if pd.isna(r): return "Other"
    r = int(r)
    if r == 1: return "White"
    if r == 2: return "Black"
    if r == 6: return "Asian"
    return "Other"

def ten_to_owner(ten):
    # TEN: 1 owned w/ mortgage, 2 owned free/clear, 3 rented
    t = pd.to_numeric(ten, errors="coerce")
    if pd.isna(t): return np.nan
    t = int(t)
    if t in (1, 2): return "owner"
    if t == 3: return "renter"
    return np.nan

def recode_marital(mar):
    # MAR: 1 = married; others = not married
    m = pd.to_numeric(mar, errors="coerce")
    return "married" if (not pd.isna(m) and int(m) == 1) else "other"

# -------------------------------------------------------------------
# Load PUMS (A+B if available)
# -------------------------------------------------------------------
person_paths = sorted([p for p in PUMS_DIR.glob("psam_pus*.csv") if p.is_file()])
house_paths  = sorted([p for p in PUMS_DIR.glob("psam_hus*.csv") if p.is_file()])

if not person_paths:
    raise SystemExit(
        "No PUMS person files found. Expected files like data/pums/psam_pusa.csv "
        "(download & unzip from the Census PUMS site)."
    )
if not house_paths:
    raise SystemExit(
        "No PUMS housing files found. Expected files like data/pums/psam_husa.csv "
        "(download & unzip from the Census PUMS site)."
    )

# Read & append (keep only needed columns)
p_list = [pd.read_csv(p, low_memory=False, usecols=[c for c in PERSON_COLS if c != "SERIALNO"] + ["SERIALNO"]) for p in person_paths]
h_list = [pd.read_csv(p, low_memory=False, usecols=HOUSE_COLS) for p in house_paths]
p = pd.concat(p_list, ignore_index=True)
h = pd.concat(h_list, ignore_index=True)

# Merge person ↔ housing
p = p.merge(h, on="SERIALNO", how="left")

# -------------------------------------------------------------------
# Build microcell features (NaN-safe)
# -------------------------------------------------------------------
p["age_bin"]    = p["AGEP"].apply(recode_age)
p["sex"]        = p["SEX"].map({1:"Male", 2:"Female"}).fillna(np.nan)
p["race_eth"]   = [recode_race_eth(r, h_) for r, h_ in zip(p["RAC1P"], p["HISP"])]
p["edu_bin"]    = p["SCHL"].apply(recode_edu)
p["income_bin"] = p["PINCP"].apply(recode_income)
p["owner"]      = p["TEN"].apply(ten_to_owner)
p["married"]    = p["MAR"].apply(recode_marital)

# Keep only schema columns and drop incomplete rows
keep = ["ST","age_bin","sex","race_eth","edu_bin","income_bin","married","owner"]
missing_any = [c for c in keep if c not in p.columns]
if missing_any:
    raise SystemExit(f"Missing expected columns after load: {missing_any}")

p = p[keep].dropna().copy()
p["n"] = 1

# Aggregate to national microcells
micro = p.groupby(keep, as_index=False)["n"].sum()

# Save
micro.to_parquet(OUT, index=False)
print(f"✓ microcells {OUT} rows={len(micro):,}")
