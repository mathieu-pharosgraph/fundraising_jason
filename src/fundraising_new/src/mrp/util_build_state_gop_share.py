#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# INPUT: block-group level presidential votes (counts, not percents)
IN_CSV = Path("voting/Block Groups/bg-2020-RLCR.csv")   # replace with 2024 file when ready
OUT_PARQ = Path("outputs/mrp/state_gop_2p.parquet")
OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)

# Load BG votes
# Expect columns at least: bg_GEOID, bg_state_fp, G20PRERTRU, G20PREDBID (your screenshot)
df = pd.read_csv(IN_CSV, dtype={"bg_state_fp":"Int64"}, low_memory=False)

# Pick columns for Trump & Biden; try 2024 first, else 2020 as fallback
CANDS = [
    ("G24PRERTRU","G24PREDBID"),  # 2024 naming (if you have it)
    ("G20PRERTRU","G20PREDBID"),  # 2020 naming (your file)
]
for tr_col, bd_col in CANDS:
    if tr_col in df.columns and bd_col in df.columns:
        trump_col, biden_col = tr_col, bd_col
        break
else:
    raise ValueError("Could not find Trump/Biden columns for 2024 or 2020 in the BG vote file.")

# Aggregate to state (sum of BG vote counts)
state = (df.groupby("bg_state_fp")[[trump_col, biden_col]]
           .sum(min_count=1)
           .reset_index()
           .rename(columns={trump_col:"trump", biden_col:"biden"}))

# Two-party GOP share
state["gop_2p"] = state["trump"] / (state["trump"] + state["biden"])

# Map state FIPS -> postal
F2P = {
  "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC",
  "12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY",
  "22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT",
  "31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH",
  "40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT",
  "50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY","72":"PR"
}
state["bg_state_fp"] = state["bg_state_fp"].astype(str).str.zfill(2)
state["state_postal"] = state["bg_state_fp"].map(F2P)

# z-score within USA (center + scale)
state["state_gop_2p_z"] = (state["gop_2p"] - state["gop_2p"].mean()) / state["gop_2p"].std(ddof=0)

state[["state_postal","gop_2p","state_gop_2p_z"]].to_parquet(OUT_PARQ, index=False)
print("✓ wrote", OUT_PARQ, "rows:", len(state))
