#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
import re

CCES = Path("/Users/mathieutrepanier/Library/CloudStorage/GoogleDrive-mathieu@xthorizon.com/My Drive/PharosGraph/R&D/Share_of_model/fundraising_participation/data/ces/raw/ces_2020.csv")
OUT  = Path("outputs/mrp/cces_2020_prepped.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------- state maps (postal <-> fips <-> name) ----------------
_fips2abbr = {
 "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
 "13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME",
 "24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
 "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
 "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI",
 "56":"WY"
}
_name2abbr = {
 "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO","connecticut":"CT","delaware":"DE",
 "district of columbia":"DC","dc":"DC","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA",
 "kansas":"KS","kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN",
 "mississippi":"MS","missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ","new mexico":"NM",
 "new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK","oregon":"OR","pennsylvania":"PA","rhode island":"RI",
 "south carolina":"SC","south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA",
 "west virginia":"WV","wisconsin":"WI","wyoming":"WY"
}

def to_state_postal_any(s):
    if s is None: return None
    t = str(s).strip()
    if not t: return None
    if len(t)==2 and t.isalpha(): return t.upper()
    # numeric fips
    f = pd.to_numeric(t, errors="coerce")
    if not pd.isna(f):
        f2 = f"{int(f):02d}"
        return _fips2abbr.get(f2)
    # full name
    return _name2abbr.get(t.lower())

# ---------------- helpers ----------------
def pick(df, *cands):
    # case-sensitive then case-insensitive
    for c in cands:
        if c and c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in low: return low[c.lower()]
    return None

def first_col_matching(df, pattern):
    rgx = re.compile(pattern, re.I)
    for c in df.columns:
        if rgx.search(c): return c
    return None

def recode_age_bin(v):
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x) or x<18 or x>105: return np.nan
    x = int(x)
    return "18-29" if x<30 else ("30-44" if x<45 else ("45-64" if x<65 else "65+"))

def recode_edu(v):
    s = str(v).strip().lower()
    if s in {"", "nan"}: return np.nan
    n = pd.to_numeric(v, errors="coerce")
    if not pd.isna(n):
        n = int(n); 
        return {1:"HS_or_less",2:"HS_or_less",3:"SomeCollege",4:"SomeCollege",5:"BAplus",6:"BAplus"}.get(n, "SomeCollege")
    if any(k in s for k in ["less than hs","no hs","high school","ged"]): return "HS_or_less"
    if any(k in s for k in ["some college","assoc","associate","2-year","two-year"]): return "SomeCollege"
    if any(k in s for k in ["bachelor","ba","bs","graduate","master","phd","jd","md","college graduate"]): return "BAplus"
    return "SomeCollege"

def recode_income_cces(val):
    # Handles numeric codes or text buckets
    s = str(val).strip().lower()
    n = pd.to_numeric(val, errors="coerce")

    # Numeric code mapping (common CCES 2020 faminc/faminc_new bins)
    # Adjust if your codebook differs:
    code_map = {
        1:"low", 2:"low", 3:"low", 4:"low",        # <35k
        5:"mid", 6:"mid", 7:"mid", 8:"mid", 9:"mid", # 35k–99k
        10:"high", 11:"high", 12:"high", 13:"high", 14:"high", 15:"high", 16:"high" # >=100k
    }
    if not pd.isna(n):
        return code_map.get(int(n), "mid")

    # Text buckets
    if any(k in s for k in ["$0", "under $10", "<", "less than", "10,000", "20,000", "30,000"]):
        return "low"
    if any(k in s for k in ["$35", "35,000", "40,000", "50,000", "60,000", "70,000", "80,000", "90,000"]):
        return "mid"
    if any(k in s for k in ["$100", "100,000", "125,000", "150,000", "200,000", "250,000", "or more", "≥", ">=100"]):
        return "high"
    return "mid"




def recode_marital_cces(v):
    n = pd.to_numeric(v, errors="coerce")
    if not pd.isna(n):
        return "married" if int(n) == 1 else "other"
    s = str(v).lower()
    if "married" in s and "not" not in s:
        return "married"
    return "other"

def recode_owner_from_ownhome(v):
    # your file: ownhome -> 1=own, 2=rent, 3=other (treat as renter)
    n = pd.to_numeric(v, errors="coerce")
    if pd.isna(n): return np.nan
    return "owner" if int(n)==1 else "renter"

def recode_race_eth(race, hisp):
    hv = str(hisp).strip().lower()
    hisp_yes = hv in {"1","yes","true","y"} or "hisp" in hv
    if hisp_yes: return "Latino"
    s = str(race).lower()
    n = pd.to_numeric(race, errors="coerce")
    if not pd.isna(n):
        n = int(n)
        if n==1: return "White"
        if n==2: return "Black"
        if n==4: return "Asian"
        if n==3: return "Latino"
        return "Other"
    if "white" in s: return "White"
    if "black" in s or "african" in s: return "Black"
    if "asian" in s: return "Asian"
    return "Other"

# -------- load
df = pd.read_csv(CCES, low_memory=False)

# -------- OUTCOME: ideology (auto-detect)
# Try common names first
ideo_col = pick(df, "cc20_340_grid","CC20_340","ideo7","IDEO7")
if ideo_col is None:
    # any column that looks like ideology?
    # prefer names containing 'ideo' or 'libcon'
    ideo_col = first_col_matching(df, r"\b(ideo|libcon)\b")
if ideo_col is None:
    # some builds use 'cc20_340' variants suffixed '_a' etc.
    ideo_col = first_col_matching(df, r"cc20[_-]?340")
if ideo_col is None:
    raise SystemExit(f"Unable to find an ideology column. "
                     f"Looked for cc20_340_grid/cc20_340/ideo7/ideo5/libcon*. "
                     f"Available columns (first 50): {list(df.columns)[:50]}")

# Detect scale (5-pt vs 7-pt) and build ideology7
ideo_vals = pd.to_numeric(df[ideo_col], errors="coerce")
unique_non_na = sorted([int(v) for v in ideo_vals.dropna().unique() if 1 <= v <= 7])
if set(unique_non_na).issubset({1,2,3,4,5}):
    # map 5->7 (coarse monotonic mapping)
    map5to7 = {1:1, 2:3, 3:4, 4:5, 5:7}
    df["ideology7"] = ideo_vals.map(lambda v: map5to7.get(int(v), np.nan) if not pd.isna(v) else np.nan)
else:
    # assume already 1..7
    df["ideology7"] = ideo_vals.clip(1,7)

# -------- OUTCOME: pid7 (if present)
pid_col = pick(df, "pid7","PID7","partyid7")
if pid_col is None:
    # fallback: any 'pid7' variant
    pid_col = first_col_matching(df, r"\bpid7\b")
if pid_col is not None:
    df["pid7"] = pd.to_numeric(df[pid_col], errors="coerce")
else:
    # allow missing pid7; we can still fit ideology + registration models
    df["pid7"] = np.nan

# -------- OUTCOME: registration (3-class) — detect from common names or numerics
reg_col = pick(df, "cc20_360","CC20_360","registration","partyreg","regparty")
if reg_col is None:
    # try any column with 'reg' and 'party' in the name
    reg_col = first_col_matching(df, r"reg.*party|party.*reg")
if reg_col is None:
    # still nothing — create Ind placeholder from voter file status (seriously last resort)
    df["registration_3"] = "Ind"
else:
    def recode_reg3(v):
        n = pd.to_numeric(v, errors="coerce")
        if not pd.isna(n):
            if int(n) == 1: return "Dem"
            if int(n) == 2: return "Rep"
            return "Ind"
        s = str(v).lower()
        if "dem" in s: return "Dem"
        if "rep" in s or "gop" in s: return "Rep"
        return "Ind"
    df["registration_3"] = df[reg_col].map(recode_reg3)

# -------- predictors
age_col    = pick(df, "age","inputage","AGE","InputAge")
birthyr    = pick(df, "birthyr","BIRTHYR")
gender_col = pick(df, "gender","sex","SEX")
race_col   = pick(df, "race","race_ethnicity","race_eth","race_7cat")
hisp_col   = pick(df, "hispanic","hisp","HISPANIC","HISP")
educ_col   = pick(df, "educ","education","EDUC")
inc_col = pick(df, "faminc","faminc_new","family_income","income","INCOME")
mar_col = pick(df, "marstat","marital","MARSTAT")
own_col    = pick(df, "ownhome","tenure","homeown","OWNHOME","TENURE")
# state: prefer 2-letter, else numeric 'inputstate', else full name
sp_two  = pick(df, "state_postal","stateabb","state_abbr","stateabbr","STATEABB")
sp_fips = pick(df, "inputstate","input_state","state_fips","STATEFIPS")
sp_name = pick(df, "state","STATE")

# age
if age_col:
    df["age_bin"] = df[age_col].map(recode_age_bin)
elif birthyr:
    df["_age_tmp"] = pd.to_numeric(df[birthyr], errors="coerce").apply(lambda y: 2020 - y if pd.notna(y) else np.nan)
    df["age_bin"] = df["_age_tmp"].map(recode_age_bin)
    df.drop(columns=["_age_tmp"], inplace=True)
else:
    df["age_bin"] = np.nan

# sex
if gender_col:
    df["sex"] = df[gender_col].replace({1:"Male",2:"Female","1":"Male","2":"Female"}).astype(str)
else:
    df["sex"] = np.nan

# race/eth
df["race_eth"] = df.apply(lambda r: recode_race_eth(r.get(race_col), r.get(hisp_col)), axis=1)

# edu
df["edu_bin"] = df[educ_col].map(recode_edu) if educ_col else np.nan

# income
df["income_bin"] = df[inc_col].map(recode_income_cces) if inc_col else "mid"
df["income_bin"] = pd.Categorical(df["income_bin"], categories=["low","mid","high"])

# married
df["married"] = df[mar_col].map(recode_marital_cces) if mar_col else "other"
df["married"] = df["married"].astype("category")

# owner (from ownhome 1/2/3; default renter)
if own_col:
    df["owner"] = df[own_col].map(recode_owner_from_ownhome)
else:
    df["owner"] = np.nan
df["owner"] = df["owner"].fillna("renter").astype("category")


# state_postal
sp = pd.Series([None]*len(df), dtype="object")
if sp_two:
    s = df[sp_two].astype(str).str.strip()
    sp = s.where(s.str.len()==2, None)
if sp.isna().all() and sp_fips:
    fips = pd.to_numeric(df[sp_fips], errors="coerce").astype("Int64")
    sp = (fips.astype("string").str.zfill(2)).map(_fips2abbr)
if sp.isna().any() and sp_name:
    miss = sp.isna()
    names = df.loc[miss, sp_name].astype(str).str.strip().str.lower()
    sp.loc[miss] = names.map(_name2abbr)
df["state_postal"] = pd.Categorical(sp).add_categories(["NA"]).fillna("NA")

# -------- final select + types
cols = ["ideology7","pid7","registration_3","state_postal",
        "age_bin","sex","race_eth","edu_bin","income_bin","married","owner"]
df_out = df[cols].copy()

for c in ["age_bin","sex","race_eth","edu_bin","income_bin","married","owner","registration_3","state_postal"]:
    df_out[c] = df_out[c].astype("category")

# drop rows missing outcomes or core sociodemo (NOT dropping state_postal)
df_out = df_out.dropna(subset=["ideology7","age_bin","sex","race_eth","edu_bin","income_bin","married","owner"])

# ------- logging
print("Detected:")
print("  ideology column:", ideo_col, "→ scaled to ideology7 (min/max):",
      pd.to_numeric(df_out["ideology7"], errors="coerce").min(), "/", pd.to_numeric(df_out["ideology7"], errors="coerce").max())
print("  pid7 column:", pid_col)
print("  registration column:", reg_col)
print("  state source:", (sp_two or sp_fips or sp_name), " → NA share:", (df_out["state_postal"]=="NA").mean().round(3))
print("  owner source:", own_col)

OUT.parent.mkdir(parents=True, exist_ok=True)
df_out.to_parquet(OUT, index=False)
print(f"✓ wrote {OUT} rows={len(df_out)}")
