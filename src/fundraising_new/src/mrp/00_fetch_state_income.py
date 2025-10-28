#!/usr/bin/env python3
import os, pandas as pd, requests
from pathlib import Path

OUT = Path("outputs/mrp/state_income_acs.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

KEY = os.getenv("CENSUS_KEY", "")
params = {"get":"NAME,B19013_001E","for":"state:*"}
if KEY: params["key"]=KEY
url = "https://api.census.gov/data/2020/acs/acs5"

r = requests.get(url, params=params, timeout=60)
r.raise_for_status()
rows = r.json()
df = pd.DataFrame(rows[1:], columns=rows[0])
df["B19013_001E"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
# map state FIPS -> postal
fips2abbr={"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY"}
df["state_fips"] = df["state"].str.zfill(2)
df["state_postal"] = df["state_fips"].map(fips2abbr)
df = df.rename(columns={"B19013_001E":"median_income"})
mu, sd = df["median_income"].mean(), df["median_income"].std()
df["state_income_z"] = (df["median_income"] - mu) / sd
df[["state_postal","median_income","state_income_z"]].to_parquet(OUT, index=False)
print("✓ wrote", OUT)
