#!/usr/bin/env python3
import pandas as pd, re, ast
from pathlib import Path

ENRICHED = Path("data/affinity/reports/topics_enriched.csv")
OUT      = Path("data/affinity/reports/topics_enriched_labels.csv")

def nkey(s): return re.sub(r"[^a-z0-9]+","", str(s).lower())

def parse_topics(s: str) -> list[str]:
    s = str(s or "").strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            # handle a weird list form like ['A' 'B']
            parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
            vals = [a or b for a,b in parts]
            if vals: return [p.strip() for p in vals if p.strip()]
    # generic delimiters
    parts = re.split(r"\s*[|;,]\s*|\s{2,}", s)
    return [p.strip() for p in parts if p.strip()]

df = pd.read_csv(ENRICHED, dtype=str, low_memory=False)
frames = []
for col in ["story_label","label","winner_label","best_label"]:
    if col in df.columns:
        t = df[[col,"standardized_topic_names"]].dropna(subset=[col]).copy()
        t["label_key"] = t[col].apply(nkey)
        frames.append(t[["label_key","standardized_topic_names"]])

m = (pd.concat(frames, ignore_index=True)
       .drop_duplicates("label_key"))

m["standardized_topic_names"] = m["standardized_topic_names"].apply(lambda s: "; ".join(parse_topics(s)))
m = m[m["standardized_topic_names"].ne("")]

OUT.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(OUT, index=False)
print("Wrote:", OUT, "rows:", len(m), "unique label_keys:", m["label_key"].nunique())
