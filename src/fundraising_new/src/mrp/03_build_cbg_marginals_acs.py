#!/usr/bin/env python3
import os, io, time, math, csv, re, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from collections import defaultdict

OUT = Path("outputs/mrp/cbg_marginals.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

CENSUS_KEY = os.getenv("CENSUS_KEY", "")
ACS_BASE = "https://api.census.gov/data/2020/acs/acs5"

# RUCA (you downloaded earlier)
TRACT_RUCA = Path("data/crosswalk/TRACT_RUCA.csv")

# ---------------------------
# Helper: RUCA loader (works for CSV or ERS zip)
# ---------------------------
def _open_ruca_bytes(path: Path) -> bytes:
    b = path.read_bytes()
    if b[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(b)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            with zf.open(names[0]) as f:
                return f.read()
    return b

def load_ruca(path: Path) -> pd.DataFrame:
    raw = _open_ruca_bytes(path).decode("utf-8", errors="replace")
    # try to sniff delimiter
    try:
        df = pd.read_csv(io.StringIO(raw), engine="python", sep=None, dtype=str)
    except Exception:
        df = pd.read_csv(io.StringIO(raw), engine="python", sep=",", dtype=str, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    tract_col = next((c for c in df.columns if re.search(r"(tract|trtid|tractce|geoid)", c, re.I)), None)
    ruca_col  = next((c for c in df.columns if re.match(r"ruca\d*$", c, re.I) or "ruca" in c.lower()), None)
    if tract_col is None or ruca_col is None:
        raise SystemExit(f"RUCA file missing columns. Got: {df.columns.tolist()}")
    df["tract11"] = df[tract_col].astype(str).str.replace(r"\D", "", regex=True).str[-11:].str.zfill(11)
    df["ruca"]    = df[ruca_col].astype(str).str.strip()
    out = df[["tract11","ruca"]].dropna()
    out = out[out["tract11"].str.fullmatch(r"\d{11}")]
    return out

if not TRACT_RUCA.exists():
    raise SystemExit("RUCA file missing. Place 2020 tract RUCA at data/crosswalk/TRACT_RUCA.csv")

RUCA = load_ruca(TRACT_RUCA)

# ---------------------------
# Census API helpers
# ---------------------------
def census_get(params: dict) -> pd.DataFrame:
    # respects throttling; returns DataFrame
    p = params.copy()
    if CENSUS_KEY:
        p["key"] = CENSUS_KEY
    for _ in range(5):
        r = requests.get(ACS_BASE, params=p, timeout=60)
        if r.status_code == 200:
            rows = r.json()
            return pd.DataFrame(rows[1:], columns=rows[0])
        if r.status_code in (429, 500, 503):
            time.sleep(1.5)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Census API failed for {params}")

def list_states() -> pd.DataFrame:
    # states incl. DC, PR (state FIPS codes)
    df = census_get({"get":"NAME", "for":"state:*"})
    return df.rename(columns={"state":"state_fips"})

# ---------------------------
# Table → bins mapping
# ---------------------------
# Sex & Age: B01001 (block-group). We'll roll to 4 age bins and 2 sex bins.
B01001_male_cols = {
    "under5":"B01001_003E","5to9":"B01001_004E","10to14":"B01001_005E","15to17":"B01001_006E",
    "18to19":"B01001_007E","20":"B01001_008E","21":"B01001_009E","22to24":"B01001_010E",
    "25to29":"B01001_011E","30to34":"B01001_012E","35to39":"B01001_013E","40to44":"B01001_014E",
    "45to49":"B01001_015E","50to54":"B01001_016E","55to59":"B01001_017E","60to61":"B01001_018E",
    "62to64":"B01001_019E","65to66":"B01001_020E","67to69":"B01001_021E","70to74":"B01001_022E",
    "75to79":"B01001_023E","80to84":"B01001_024E","85plus":"B01001_025E"
}
B01001_female_cols = {
    "under5":"B01001_027E","5to9":"B01001_028E","10to14":"B01001_029E","15to17":"B01001_030E",
    "18to19":"B01001_031E","20":"B01001_032E","21":"B01001_033E","22to24":"B01001_034E",
    "25to29":"B01001_035E","30to34":"B01001_036E","35to39":"B01001_037E","40to44":"B01001_038E",
    "45to49":"B01001_039E","50to54":"B01001_040E","55to59":"B01001_041E","60to61":"B01001_042E",
    "62to64":"B01001_043E","65to66":"B01001_044E","67to69":"B01001_045E","70to74":"B01001_046E",
    "75to79":"B01001_047E","80to84":"B01001_048E","85plus":"B01001_049E"
}
B01001_all_cols = list(B01001_male_cols.values()) + list(B01001_female_cols.values())

def age_bucket(cols: dict, row: pd.Series, lo:int, hi:int) -> float:
    # map detailed B01001 bins to our coarse 18-29, 30-44, 45-64, 65+
    # we just sum bins whose label falls in the coarse range
    def pick(keys):
        return sum(float(row.get(cols[k], 0) or 0) for k in keys)
    if (lo,hi)==(18,29):
        return pick(["18to19","20","21","22to24","25to29"])
    if (lo,hi)==(30,44):
        return pick(["30to34","35to39","40to44"])
    if (lo,hi)==(45,64):
        return pick(["45to49","50to54","55to59","60to61","62to64"])
    if (lo,hi)==(65,200):
        return pick(["65to66","67to69","70to74","75to79","80to84","85plus"])
    return 0.0

# Race/Hispanic: use B03002 (block-group): Hispanic + race for Not Hispanic
# We'll output categories: White, Black, Latino, Asian, Other
B03002_vars = [
    "B03002_001E", # total
    "B03002_003E", # White alone, not Hispanic
    "B03002_004E", # Black alone, not Hispanic
    "B03002_006E", # Asian alone, not Hispanic
    "B03002_012E", # Hispanic or Latino
]

# Education: B15003 (educ attainment 25+) — available at block-group
# We’ll collapse to HS_or_less / SomeCollege / BAplus
#  B15003_001E total 25+
#  <=HS bins: 2..16 ; SomeCollege/Assoc: 17..20 ; BA+: 21..25
B15003_bins = {
    "HS_or_less":  [f"B15003_{i:03d}E" for i in range(2, 17)],   # 002–016
    "SomeCollege": [f"B15003_{i:03d}E" for i in range(17, 21)],  # 017–020
    "BAplus":      [f"B15003_{i:03d}E" for i in range(21, 26)],  # 021–025
}
B15003_vars = ["B15003_001E"] + sorted(set(sum(B15003_bins.values(), [])))

# Tenure: B25003 (occupied housing units) — owner vs renter at block-group
B25003_vars = ["B25003_001E","B25003_002E","B25003_003E"] # total, owner, renter

# Marital status: B12001 often **not** at block-group for small areas; use **tract**
B12001_vars = [f"B12001_{i:03d}E" for i in range(1, 6)]          # 001–005

# We'll approximate "married" share as (married) / total for tract and allocate to its BGs by population weight

# Income: B19001 (household income) commonly tract-level for stable estimates
# We'll collapse to low (<35k), mid (35k-100k), high (>=100k)
B19001_bins = {
    "low":  [f"B19001_{i:03d}E" for i in range(2, 8)],           # 002–007
    "mid":  [f"B19001_{i:03d}E" for i in range(8, 14)],          # 008–013
    "high": [f"B19001_{i:03d}E" for i in range(14, 18)],         # 014–017
}
B19001_vars = ["B19001_001E"] + sorted(set(sum(B19001_bins.values(), [])))

# ---------------------------
# Pull helpers
# ---------------------------
def pull_blockgroup(vars_list, state_fips):
    get = ",".join(["NAME"] + vars_list)
    df = census_get({"get": get, "for": "block group:*", "in": f"state:{state_fips} county:*"})
    # construct 12-digit CBG: state(2)+county(3)+tract(6)+bg(1)
    df["cbg_id"] = (
        df["state"].str.zfill(2)
        + df["county"].str.zfill(3)
        + df["tract"].str.zfill(6)
        + df["block group"].str.zfill(1)
    )
    return df

def pull_tract(vars_list, state_fips):
    get = ",".join(["NAME"] + vars_list)
    df = census_get({"get": get, "for": "tract:*", "in": f"state:{state_fips} county:*"})
    df["tract11"] = df["state"].str.zfill(2) + df["county"].str.zfill(3) + df["tract"].str.zfill(6)
    return df

# ---------------------------
# Build per-state marginals
# ---------------------------
def build_state(state_fips: str) -> pd.DataFrame:
    out_rows = []

    # 1) Age/Sex at block-group: B01001
    bg1 = pull_blockgroup(B01001_all_cols, state_fips)
    # sex totals
    bg1["male_total"]   = bg1[B01001_male_cols.values()].astype(float).sum(axis=1)
    bg1["female_total"] = bg1[B01001_female_cols.values()].astype(float).sum(axis=1)
    # age bins per sex
    for lo,hi,label in [(18,29,"18-29"), (30,44,"30-44"), (45,64,"45-64"), (65,200,"65+")]:
        bg1[f"male_{label}"] = bg1.apply(lambda r: age_bucket(B01001_male_cols, r, lo, hi), axis=1)
        bg1[f"female_{label}"]= bg1.apply(lambda r: age_bucket(B01001_female_cols, r, lo, hi), axis=1)
    # write long rows
    for _,r in bg1.iterrows():
        cbg = r["cbg_id"]
        # sex
        out_rows.append({"cbg_id":cbg, "dim":"sex", "category":"Male",   "n": float(r["male_total"])})
        out_rows.append({"cbg_id":cbg, "dim":"sex", "category":"Female", "n": float(r["female_total"])})
        # age (sum male+female)
        for label in ["18-29","30-44","45-64","65+"]:
            n = float(r[f"male_{label}"])+float(r[f"female_{label}"])
            out_rows.append({"cbg_id":cbg, "dim":"age_bin","category":label,"n":n})

    # 2) Race/Ethnicity at block-group: B03002
    bg2 = pull_blockgroup(B03002_vars, state_fips).astype({v:"float" for v in B03002_vars})
    for _,r in bg2.iterrows():
        cbg = r["cbg_id"]
        total = r["B03002_001E"] or 0.0
        nh_white = r["B03002_003E"] or 0.0
        nh_black = r["B03002_004E"] or 0.0
        nh_asian = r["B03002_006E"] or 0.0
        latino   = r["B03002_012E"] or 0.0
        other = max(total - (nh_white+nh_black+nh_asian+latino), 0.0)
        out_rows += [
            {"cbg_id":cbg,"dim":"race_eth","category":"White","n":float(nh_white)},
            {"cbg_id":cbg,"dim":"race_eth","category":"Black","n":float(nh_black)},
            {"cbg_id":cbg,"dim":"race_eth","category":"Asian","n":float(nh_asian)},
            {"cbg_id":cbg,"dim":"race_eth","category":"Latino","n":float(latino)},
            {"cbg_id":cbg,"dim":"race_eth","category":"Other","n":float(other)},
        ]

    # 3) Education at block-group: B15003 (25+)
    bg3 = pull_blockgroup(B15003_vars, state_fips).astype({v:"float" for v in B15003_vars})
    for _,r in bg3.iterrows():
        cbg = r["cbg_id"]
        # collapse
        hs  = sum(r[v] for v in B15003_bins["HS_or_less"]   if v in r)
        some= sum(r[v] for v in B15003_bins["SomeCollege"]  if v in r)
        ba  = sum(r[v] for v in B15003_bins["BAplus"]       if v in r)
        out_rows += [
            {"cbg_id":cbg,"dim":"edu_bin","category":"HS_or_less","n":float(hs)},
            {"cbg_id":cbg,"dim":"edu_bin","category":"SomeCollege","n":float(some)},
            {"cbg_id":cbg,"dim":"edu_bin","category":"BAplus","n":float(ba)},
        ]

    # 4) Tenure at block-group: B25003 owner vs renter
    bg4 = pull_blockgroup(B25003_vars, state_fips).astype({v:"float" for v in B25003_vars})
    for _,r in bg4.iterrows():
        cbg = r["cbg_id"]
        own = r["B25003_002E"] or 0.0
        rent= r["B25003_003E"] or 0.0
        out_rows += [
            {"cbg_id":cbg,"dim":"owner","category":"owner","n":float(own)},
            {"cbg_id":cbg,"dim":"owner","category":"renter","n":float(rent)},
        ]

    # 5) Marital status at TRACT: B12001 → allocate to BG by pop share
    tr1 = pull_tract(B12001_vars, state_fips).astype({v:"float" for v in B12001_vars})
    # very rough married count: male+female married (tables differ by sex); many analysts use B12001_003E+B12001_008E
    tr1["married"] = tr1.get("B12001_003E",0.0) + tr1.get("B12001_008E",0.0)
    tr1 = tr1[["tract11","B12001_001E","married"]].rename(columns={"B12001_001E":"tot"})
    # block-group population from age table as weight:
    pop_bg = bg1[["cbg_id"] + list(B01001_male_cols.values()) + list(B01001_female_cols.values())].copy()
    pop_bg["pop"] = pop_bg.drop(columns=["cbg_id"]).astype(float).sum(axis=1)
    pop_bg["tract11"] = pop_bg["cbg_id"].str.slice(0,11)
    # compute each BG's share within tract
    tot_by_tract = pop_bg.groupby("tract11", as_index=False)["pop"].sum().rename(columns={"pop":"tract_pop"})
    pop_bg = pop_bg.merge(tot_by_tract, on="tract11", how="left")
    pop_bg["w"] = np.where(pop_bg["tract_pop"]>0, pop_bg["pop"]/pop_bg["tract_pop"], 0.0)

    tr1 = tr1.merge(pop_bg[["cbg_id","tract11","w"]], on="tract11", how="right")
    tr1["married_alloc"] = tr1["married"] * tr1["w"]
    tr1["other_alloc"]   = (tr1["tot"] - tr1["married"]) * tr1["w"]
    for _,r in tr1.iterrows():
        out_rows += [
            {"cbg_id":r["cbg_id"], "dim":"married","category":"married","n": float(max(r["married_alloc"],0.0))},
            {"cbg_id":r["cbg_id"], "dim":"married","category":"other",  "n": float(max(r["other_alloc"],0.0))},
        ]

    # 6) Income at TRACT: B19001 → allocate to BG by households proxy (tenure total)
    tr2 = pull_tract(B19001_vars, state_fips).astype({v:"float" for v in B19001_vars})
    tr2["low"]  = tr2[B19001_bins["low"]].sum(axis=1)
    tr2["mid"]  = tr2[B19001_bins["mid"]].sum(axis=1)
    tr2["high"] = tr2[B19001_bins["high"]].sum(axis=1)
    tr2 = tr2[["tract11","low","mid","high","B19001_001E"]].rename(columns={"B19001_001E":"hh_tot"})
    # BG household proxy: B25003 total occupied units owner+rent
    hh_bg = bg4[["cbg_id","B25003_002E","B25003_003E"]].copy()
    hh_bg["hh"] = hh_bg[["B25003_002E","B25003_003E"]].astype(float).sum(axis=1)
    hh_bg["tract11"] = hh_bg["cbg_id"].str.slice(0,11)
    hh_tot = hh_bg.groupby("tract11", as_index=False)["hh"].sum().rename(columns={"hh":"tract_hh"})
    hh_bg = hh_bg.merge(hh_tot, on="tract11", how="left")
    hh_bg["w"] = np.where(hh_bg["tract_hh"]>0, hh_bg["hh"]/hh_bg["tract_hh"], 0.0)

    tr2 = tr2.merge(hh_bg[["cbg_id","tract11","w"]], on="tract11", how="right")
    for cat in ["low","mid","high"]:
        tr2[f"{cat}_alloc"] = tr2[cat] * tr2["w"]

    for _,r in tr2.iterrows():
        cbg = r["cbg_id"]
        out_rows += [
            {"cbg_id":cbg,"dim":"income_bin","category":"low", "n": float(max(r["low_alloc"],0.0))},
            {"cbg_id":cbg,"dim":"income_bin","category":"mid", "n": float(max(r["mid_alloc"],0.0))},
            {"cbg_id":cbg,"dim":"income_bin","category":"high","n": float(max(r["high_alloc"],0.0))},
        ]

    # return state frame
    return pd.DataFrame(out_rows)

# ---------------------------
# Run for all states (can be slow) — start with a subset if testing
# ---------------------------
states = list_states()["state_fips"].tolist()
# For a quick smoke test, uncomment:
# states = ["06","08","36"]  # CA, CO, NY

frames = []
for s in states:
    print(f"Building ACS marginals for state {s} ...")
    try:
        frames.append(build_state(s))
    except Exception as e:
        print(f"WARNING: state {s} failed: {e}")

cbg_marg = pd.concat(frames, ignore_index=True)

# Attach RUCA (tract11 from cbg_id)
cbg_marg["tract11"] = cbg_marg["cbg_id"].str.slice(0,11)
cbg_marg = cbg_marg.merge(RUCA, on="tract11", how="left").drop(columns=["tract11"])

# Save
cbg_marg.to_parquet(OUT, index=False)
print(f"✓ wrote {OUT} rows= {len(cbg_marg):,}  (unique CBGs: {cbg_marg['cbg_id'].nunique():,})")
