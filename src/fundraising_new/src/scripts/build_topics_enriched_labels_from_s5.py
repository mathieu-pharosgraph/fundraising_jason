#!/usr/bin/env python3
import re, json, ast
import pandas as pd
from pathlib import Path

FD_PATH   = "data/affinity/topic_affinity_by_cbg_featuredot.parquet"   # for coverage check (optional)
S5_PATH   = "data/topics/merged_data_with_topics.parquet"
ENRICHED  = "data/affinity/reports/topics_enriched.csv"                # augmentation source
OUT_CSV   = "data/affinity/reports/topics_enriched_labels.csv"

topic_id_to_name = {
    0:"Non-Political / Other",1:"Abortion & Reproductive Rights",2:"Immigration Policy & Enforcement",
    3:"Trump Administration & Policy Agenda",4:"Election Integrity & Voting Rights",
    5:"Healthcare Policy (ACA, Medicare, Medicaid)",6:"Supreme Court & Judicial Affairs",
    7:"Gun Policy & Violence",8:"LGBTQ+ Rights",9:"Economic Policy & indicators",
    10:"Student Debt & Loan Forgiveness",11:"Vaccines & Public Health",
    12:"Climate Change & Environmental Policy",13:"Foreign Policy & National Security",
    14:"Ukraine-Russia War",15:"Israel-Palestine Conflict",16:"Civil Liberties & Free Speech",
    17:"Law Enforcement & Policing",18:"Congressional Dynamics & Legislation",
    19:"Corporate Accountability & Business",20:"Technology & AI Regulation",
    21:"Labor & Workers' Rights",22:"Education Policy",23:"Social Security & Welfare Programs",
    24:"Media & Journalism",25:"Entertainment & Culture",26:"Ethics & Corruption Scandals",
    27:"Extremism & Domestic Threats",28:"Censorship & Misinformation",
    29:"Federal Agency Oversight",30:"State vs. Federal Power",31:"Refugee & Asylum Seeker Crisis",
    32:"International Human Rights",33:"Campaign Finance & Politics",
    34:"Historical Legacy & Commemoration",35:"Crime & Public Safety",
    36:"Housing and Affordability",37:"Opioid Crisis & Substance Abuse",38:"Taxation & Fiscal Policy",
}
name_to_id = {v:k for k,v in topic_id_to_name.items()}

def nkey(s:str)->str: return re.sub(r"[^a-z0-9]+","", str(s).lower())

def _to_ids(x):
    """Parse topic IDs from list/JSON/array-ish/single int."""
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, (list, tuple)): return [int(i) for i in x if str(i).isdigit()]
    s = str(x).strip()
    if not s: return []
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, tuple)):
            return [int(i) for i in obj if str(i).isdigit()]
    except Exception:
        pass
    parts = re.split(r"[,\s]+", re.sub(r"[\[\]'\"']", " ", s))
    return [int(p) for p in parts if p.isdigit()]

def parse_names(val):
    """Parse names from JSON/list or delimited string."""
    if val is None: return []
    if isinstance(val, (list,tuple)): return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s: return []
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, (list,tuple)):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # split on ; | , (and collapse multi-spaces)
    return [p.strip() for p in re.split(r"\s*;\s*|\s*\|\s*|,\s*", s) if p.strip()]

def ids_to_names(ids):  # -> distinct canonical names (order preserved)
    out, seen = [], set()
    for i in ids:
        nm = topic_id_to_name.get(int(i))
        if nm and nm not in seen:
            out.append(nm); seen.add(nm)
    return out

def main():
    # 1) S5 canonical
    s5 = pd.read_parquet(S5_PATH)
    s5["label_key"] = s5.get("label", s5.get("story_label","")).astype(str).map(nkey)
    if "standardized_topic_ids" in s5.columns:
        s5["topic_ids"] = s5["standardized_topic_ids"].apply(_to_ids)
        s5_names = (s5.groupby("label_key", as_index=False)["topic_ids"]
                      .apply(lambda ser: ids_to_names([i for sub in ser for i in (sub or [])])))
        s5_names["standardized_topic_names"] = s5_names["topic_ids"].apply(lambda lst: "; ".join(lst) if lst else "")
        s5_names = s5_names[["label_key","standardized_topic_names"]]
    elif "standardized_topic_names" in s5.columns:
        s5_tmp = s5[["label_key","standardized_topic_names"]].copy()
        s5_tmp["standardized_topic_names"] = s5_tmp["standardized_topic_names"].apply(
            lambda v: "; ".join([nm for nm in parse_names(v) if nm in name_to_id]))
        s5_names = (s5_tmp.dropna(subset=["label_key"])
                         .groupby("label_key", as_index=False)["standardized_topic_names"]
                         .apply(lambda ser: "; ".join([x for x in "; ".join(ser).split("; ") if x])))
    else:
        s5_names = pd.DataFrame(columns=["label_key","standardized_topic_names"])

    # 2) ENRICHED augmentation (fill missing label_keys only)
    if Path(ENRICHED).exists():
        en = pd.read_csv(ENRICHED, dtype=str, low_memory=False)
        labcol = "story_label" if "story_label" in en.columns else ("label" if "label" in en.columns else None)
        if labcol and "standardized_topic_names" in en.columns:
            en["label_key"] = en[labcol].astype(str).map(nkey)
            en_aug = (en[["label_key","standardized_topic_names"]]
                        .dropna(subset=["label_key","standardized_topic_names"])
                        .drop_duplicates("label_key"))
        else:
            en_aug = pd.DataFrame(columns=["label_key","standardized_topic_names"])
    else:
        en_aug = pd.DataFrame(columns=["label_key","standardized_topic_names"])

    # prefer S5; fill missing with enriched
    out = s5_names.copy()
    if not en_aug.empty:
        out = out.merge(en_aug, on="label_key", how="outer", suffixes=("","__en"))
        out["standardized_topic_names"] = out["standardized_topic_names"].fillna(out["standardized_topic_names__en"])
        out = out.drop(columns=["standardized_topic_names__en"], errors="ignore")

    # keep canonical rows only (non-empty)
    out = out[out["standardized_topic_names"].astype(str).str.strip().ne("")]
    out = out.drop_duplicates("label_key")

    # write
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # optional coverage report vs feature-dot
    try:
        fd = pd.read_parquet(FD_PATH, columns=["label"])
        fd["label_key"] = fd["label"].astype(str).map(nkey)
        covered = fd["label_key"].isin(out["label_key"]).mean()*100.0
        print(f"wrote {OUT_CSV} rows={len(out)} | coverage_vs_featuredot={covered:.1f}%")
    except Exception:
        print(f"wrote {OUT_CSV} rows={len(out)}")

if __name__ == "__main__":
    main()
