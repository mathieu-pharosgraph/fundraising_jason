import pandas as pd, re, os
from pathlib import Path

def nkey(s): return re.sub(r"[^a-z0-9]+","", str(s).lower())

FD = "data/affinity/topic_affinity_by_cbg_featuredot.parquet"
EN = "data/affinity/reports/topics_enriched.csv"
OUT = "data/affinity/reports/topics_enriched_labels.csv"

# 1) Collect all label_keys present in feature-dot
cols = ["label_key","label"]
want = []
try:
    df = pd.read_parquet(FD, columns=cols)
except Exception:
    # older featuredot without label_key -> compute
    df = pd.read_parquet(FD, columns=["label"])
    df["label_key"] = df["label"].astype(str).map(nkey)
df = df.dropna(subset=["label"]).drop_duplicates("label_key")
fd_map = df[["label_key","label"]].rename(columns={"label":"fallback_label"})

# 2) Bring your existing mapping if present
if Path(EN).exists():
    en = pd.read_csv(EN, dtype=str, low_memory=False)
    labcol = "story_label" if "story_label" in en.columns else ("label" if "label" in en.columns else None)
    if labcol:
        en["label_key"] = en[labcol].astype(str).map(nkey)
        en = en[["label_key","standardized_topic_names"]].drop_duplicates("label_key")
    else:
        en = pd.DataFrame(columns=["label_key","standardized_topic_names"])
else:
    en = pd.DataFrame(columns=["label_key","standardized_topic_names"])

# 3) Merge & fallback to raw label text when mapping is missing
m = fd_map.merge(en, on="label_key", how="left")
m["standardized_topic_names"] = m["standardized_topic_names"].where(
    m["standardized_topic_names"].notna(),
    m["fallback_label"].astype(str)
)
m = m[["label_key","standardized_topic_names"]].drop_duplicates("label_key")

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
m.to_csv(OUT, index=False)
print("✓ wrote", OUT, "rows=", len(m))
