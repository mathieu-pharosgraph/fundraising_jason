#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

MARGS = Path("outputs/mrp/cbg_marginals.parquet")  # has cbg_id + RUCA + B01001-derived pop we can reconstruct
OUT   = Path("outputs/mrp/state_urban_index_acs.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Load CBG marginals with RUCA
m = pd.read_parquet(MARGS)

# Build a CBG population proxy from B01001 totals:
# Sum all B01001 male+female bins per cbg_id (we created these earlier when building age/sex marginals)
# If you didn’t keep a ready total, approximate pop as the sum of the age_bin categories per CBG.
pop = (
    m[m["dim"]=="age_bin"]
      .groupby(["cbg_id"], as_index=False)["n"]
      .sum()
      .rename(columns={"n":"cbg_pop"})
)

# RUCA -> urban/suburban/rural
# RUCA categories (1–10):
#   Urban:    1-3
#   Suburban: 4-6
#   Rural:    7-10
# m already has RUCA merged; rename to ruca
if "ruca" not in m.columns:
    raise SystemExit("cbg_marginals.parquet missing 'ruca' column. Re-run step 03 to attach RUCA.")

cbg = pop.copy()
cbg = cbg.merge(
    m[["cbg_id","ruca"]].drop_duplicates("cbg_id"),
    on="cbg_id", how="left"
)
cbg["ruca_num"] = pd.to_numeric(cbg["ruca"], errors="coerce")
def ruca_to_class(v):
    if pd.isna(v): return "unknown"
    v = int(v)
    if 1 <= v <= 3:  return "urban"
    if 4 <= v <= 6:  return "suburban"
    if 7 <= v <= 10: return "rural"
    return "unknown"
cbg["urban_cat"] = cbg["ruca_num"].map(ruca_to_class)

# Map CBG->state (first two digits of 12-digit CBG)
cbg["state_fips"] = cbg["cbg_id"].str.slice(0,2)
fips2abbr = {"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
             "13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME",
             "24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
             "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
             "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI",
             "56":"WY"}
cbg["state_postal"] = cbg["state_fips"].map(fips2abbr)

# Compute state shares (population-weighted)
grp = cbg.groupby(["state_postal","urban_cat"], as_index=False)["cbg_pop"].sum()
tot = grp.groupby("state_postal", as_index=False)["cbg_pop"].sum().rename(columns={"cbg_pop":"state_pop"})
grp = grp.merge(tot, on="state_postal", how="left")
grp["share"] = grp["cbg_pop"] / grp["state_pop"]

wide = grp.pivot(index="state_postal", columns="urban_cat", values="share").fillna(0.0).reset_index()
for c in ["urban","suburban","rural","unknown"]:
    if c not in wide.columns: wide[c]=0.0

# Build an index: urban share minus rural share (bounded ~[-1,1])
wide["urban_index"] = wide["urban"] - wide["rural"]

# z-score
mu = wide["urban_index"].mean()
sd = wide["urban_index"].std(ddof=0) or 1.0
wide["state_urban_z"] = (wide["urban_index"] - mu) / sd

wide[["state_postal","state_urban_z","urban","suburban","rural"]].to_parquet(OUT, index=False)
print("✓ wrote", OUT, "rows=", len(wide))
