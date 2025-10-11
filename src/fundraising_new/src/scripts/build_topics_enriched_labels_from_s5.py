#!/usr/bin/env python3
import re, json
import pandas as pd
from pathlib import Path

FD_PATH   = "data/affinity/topic_affinity_by_cbg_featuredot.parquet"
S5_PATH   = "data/topics/merged_data_with_topics.parquet"   # from s5_stories_to_topics.py
OUT_CSV   = "data/affinity/reports/topics_enriched_labels.csv"

# --- your canonical taxonomy ---
topic_id_to_name = {
    0: "Non-Political / Other",
    1: "Abortion & Reproductive Rights",
    2: "Immigration Policy & Enforcement",
    3: "Trump Administration & Policy Agenda",
    4: "Election Integrity & Voting Rights",
    5: "Healthcare Policy (ACA, Medicare, Medicaid)",
    6: "Supreme Court & Judicial Affairs",
    7: "Gun Policy & Violence",
    8: "LGBTQ+ Rights",
    9: "Economic Policy & indicators",
    10: "Student Debt & Loan Forgiveness",
    11: "Vaccines & Public Health",
    12: "Climate Change & Environmental Policy",
    13: "Foreign Policy & National Security",
    14: "Ukraine-Russia War",
    15: "Israel-Palestine Conflict",
    16: "Civil Liberties & Free Speech",
    17: "Law Enforcement & Policing",
    18: "Congressional Dynamics & Legislation",
    19: "Corporate Accountability & Business",
    20: "Technology & AI Regulation",
    21: "Labor & Workers' Rights",
    22: "Education Policy",
    23: "Social Security & Welfare Programs",
    24: "Media & Journalism",
    25: "Entertainment & Culture",
    26: "Ethics & Corruption Scandals",
    27: "Extremism & Domestic Threats",
    28: "Censorship & Misinformation",
    29: "Federal Agency Oversight",
    30: "State vs. Federal Power",
    31: "Refugee & Asylum Seeker Crisis",
    32: "International Human Rights",
    33: "Campaign Finance & Politics",
    34: "Historical Legacy & Commemoration",
    35: "Crime & Public Safety",
    36: "Housing and Affordability",
    37: "Opioid Crisis & Substance Abuse",
    38: "Taxation & Fiscal Policy",
}

def nkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+","", str(s).lower())

def _to_ids(x):
    """Parse topic IDs from various formats (list, JSON string, numpy-ish string, single int)."""
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, (list, tuple)): return [int(i) for i in x if str(i).isdigit()]
    s = str(x).strip()
    if not s: return []
    # JSON list?
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, tuple)):
            return [int(i) for i in obj if str(i).isdigit()]
    except Exception:
        pass
    # numpy array-ish "['A' 'B']" → split on quotes or spaces; try ints only
    parts = re.split(r"[,\s]+", re.sub(r"[\[\]'\"']", " ", s))
    ids = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            ids.append(int(p))
    return ids

def ids_to_names(ids):
    names = [topic_id_to_name.get(int(i)) for i in ids if int(i) in topic_id_to_name]
    # dedupe, keep order
    out, seen = [], set()
    for n in names:
        if n and n not in seen:
            out.append(n); seen.add(n)
    return out

def main():
    # 1) feature-dot labels (source of truth for what must be mapped)
    try:
        fd = pd.read_parquet(FD_PATH, columns=["label_key","label"])
        if "label_key" not in fd.columns:
            # older runs — derive it
            fd["label_key"] = fd["label"].astype(str).map(nkey)
    except Exception:
        fd = pd.read_parquet(FD_PATH, columns=["label"])
        fd["label_key"] = fd["label"].astype(str).map(nkey)

    fd = fd.dropna(subset=["label"]).drop_duplicates("label_key")[["label_key","label"]]
    fd = fd.rename(columns={"label":"fallback_label"})

    # 2) s5 topics file (should contain 'label' and either topic IDs or names)
    s5 = pd.read_parquet(S5_PATH)
    # prefer standardized_topic_ids if present; else try standardized_topic_names and map back via dict (best effort)
    have_ids = "standardized_topic_ids" in s5.columns
    have_names = "standardized_topic_names" in s5.columns

    s5 = s5.copy()
    s5["label_key"] = s5.get("label", s5.get("story_label","")).astype(str).map(nkey)

    if have_ids:
        s5["topic_ids"] = s5["standardized_topic_ids"].apply(_to_ids)
    elif have_names:
        # try to map names to IDs by reverse dict (best effort)
        name_to_id = {v: k for k, v in topic_id_to_name.items()}
        def _names_to_ids(val):
            if val is None: return []
            # handle JSON/"['A','B']"/"A;B" variants
            if isinstance(val, (list, tuple)):
                names = [str(x).strip() for x in val]
            else:
                s = str(val)
                if s.startswith("["):
                    try:
                        arr = json.loads(s)
                        names = [str(x).strip() for x in (arr if isinstance(arr, (list,tuple)) else [s])]
                    except Exception:
                        names = re.split(r"\s*;\s*|\s*\|\s*|,\s*", s)
                else:
                    names = re.split(r"\s*;\s*|\s*\|\s*|,\s*", s)
            ids = []
            for nm in names:
                nm = nm.strip()
                if nm in name_to_id:
                    ids.append(name_to_id[nm])
            return ids
        s5["topic_ids"] = s5["standardized_topic_names"].apply(_names_to_ids)
    else:
        s5["topic_ids"] = [[]]

    # Reduce to one row per label_key with a clean semicolon-separated name list
    agg = (s5.groupby("label_key", as_index=False)["topic_ids"]
             .apply(lambda ser: ids_to_names([i for sub in ser for i in (sub or [])])))
    agg["standardized_topic_names"] = agg["topic_ids"].apply(lambda lst: "; ".join(lst) if lst else "")
    agg = agg[["label_key","standardized_topic_names"]]

    # 3) Left-join to every label_key in feature-dot; fallback to raw label when missing
    out = fd.merge(agg, on="label_key", how="left")

    def _fallback(val, fb):
        # treat None/NaN/"nan"/"" as missing
        if val is None:
            return fb
        s = str(val).strip()
        if s == "" or s.lower() == "nan":
            return fb
        return s

    out["standardized_topic_names"] = [
        _fallback(val, fb) for val, fb in zip(out.get("standardized_topic_names"), out["fallback_label"])
    ]
    out = out[["label_key","standardized_topic_names"]].drop_duplicates("label_key")


if __name__ == "__main__":
    main()
