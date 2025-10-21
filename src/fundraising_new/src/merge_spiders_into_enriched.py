#!/usr/bin/env python3
import argparse, re, ast, json
import pandas as pd
from pathlib import Path

def nkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+","", str(s).lower())

def pct(series: pd.Series) -> float:
    s = series
    if s.dtype == "O":
        return float(s.notna().mean() * 100.0)
    return float(pd.to_numeric(s, errors="coerce").notna().mean() * 100.0)

def to_list(x):
    if isinstance(x, list): return x
    s = str(x or "").strip()
    if not s: return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s); return obj if isinstance(obj, list) else [obj]
            except Exception: pass
    return [t.strip() for t in re.split(r"\s*[;|,]\s*", s) if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)    # data/affinity/reports/topics_enriched.csv
    ap.add_argument("--spiders",   required=True)   # data/affinity/reports/topics_enriched_spiders.csv
    ap.add_argument("--out",       required=True)
    args = ap.parse_args()

    print(f"[merge_spiders_into_enriched] Starting merge for both fundraising and voting...")

    # ------- load enriched (BOTH FUNDRAISING + VOTING)
    en = pd.read_csv(args.enriched)
    print(f"[merge_spiders_into_enriched] Loaded enriched: {len(en)} rows, columns: {list(en.columns)}")
    
    en["period_norm"] = pd.to_datetime(en.get("period",""), errors="coerce").dt.date.astype("string")
    lab_en = "story_label" if "story_label" in en.columns else ("label" if "label" in en.columns else None)
    en["label_key"] = en[lab_en].astype(str).apply(nkey) if lab_en else ""

    # Clean standardized topics (make sure it's "a; b; c" not [..] or "nan"
    if "standardized_topic_names" in en.columns:
        s = en["standardized_topic_names"].astype(str)
        s = s.where(~s.str.lower().eq("nan"), "")
        en["standardized_topic_names"] = s.apply(lambda v: "; ".join(to_list(v)))

    # ------- load spiders
    sp = pd.read_csv(args.spiders)
    print(f"[merge_spiders_into_enriched] Loaded spiders: {len(sp)} rows, columns: {list(sp.columns)}")
    
    per_col = "period_norm" if "period_norm" in sp.columns else ("period" if "period" in sp.columns else None)
    if not per_col or "label_key" not in sp.columns:
        print("[merge] spiders missing keys; abort."); en.to_csv(args.out, index=False); return

    sp["period_norm"] = pd.to_datetime(sp[per_col], errors="coerce").dt.date.astype("string")
    
    # Define target columns for BOTH fundraising and voting
    # Numeric columns (emotions, moral foundations, CTA strength)
    base_num_prefixes = ["emo_", "mf_", "cta_ask_strength"]
    base_txt_columns = ["cta_ask_type", "cta_copy", "heroes", "villains", "victims", "antiheroes"]
    
    # Generate fundraising and voting versions
    fundraising_num = [f"fundraising_{c}" for c in base_num_prefixes]
    voting_num = [f"voting_{c}" for c in base_num_prefixes]
    fundraising_txt = [f"fundraising_{c}" for c in base_txt_columns]
    voting_txt = [f"voting_{c}" for c in base_txt_columns]
    
    # Combine all targets
    targets_num = fundraising_num + voting_num
    targets_txt = fundraising_txt + voting_txt
    
    print(f"[merge_spiders_into_enriched] Looking for {len(targets_num)} numeric columns and {len(targets_txt)} text columns")
    
    # strings -> NaN for empties - process BOTH fundraising and voting columns
    for c in sp.columns:
        if any(c.startswith(prefix) for prefix in (["emo_", "mf_", "cta_ask_strength", 
                                                   "fundraising_emo_", "fundraising_mf_", "fundraising_cta_ask_strength",
                                                   "voting_emo_", "voting_mf_", "voting_cta_ask_strength"])):
            sp[c] = pd.to_numeric(sp[c], errors="coerce")

    # Filter to only include columns that actually exist in spiders data
    targets_num = [c for c in targets_num if c in sp.columns]
    targets_txt = [c for c in targets_txt if c in sp.columns]
    
    print(f"[merge_spiders_into_enriched] Found {len(targets_num)} numeric columns in spiders: {targets_num}")
    print(f"[merge_spiders_into_enriched] Found {len(targets_txt)} text columns in spiders: {targets_txt}")

    # ------- diagnostics (before)
    print("BEFORE coverage:")
    all_targets = targets_num + targets_txt
    for c in all_targets:
        if c in en.columns:
            print(f"  {c:>35}: {pct(en[c]):6.1f}%")
        else:
            print(f"  {c:>35}: (missing in enriched)")

    # ------- strict merge by (period_norm, label_key)
    keep_cols = ["period_norm","label_key"] + targets_num + targets_txt
    spm = sp[keep_cols].drop_duplicates(["period_norm","label_key"], keep="last")
    out = en.merge(spm.rename(columns={c:f"{c}__sp" for c in targets_num + targets_txt}),
                   on=["period_norm","label_key"], how="left")

    # fill numerics first - handle BOTH fundraising and voting
    for c in targets_num:
        spc = f"{c}__sp"
        # ensure the column exists in out
        if c not in out.columns:
            out[c] = pd.Series([pd.NA]*len(out))
        out[c] = pd.to_numeric(out[c],  errors="coerce")
        out[spc] = pd.to_numeric(out[spc], errors="coerce")
        out[c] = out[c].where(out[c].notna(), out[spc])
        out.drop(columns=[spc], inplace=True, errors="ignore")

    # fill texts only if blank - handle BOTH fundraising and voting
    def blank(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().isin(["", "nan", "None", "—"])

    for c in targets_txt:
        spc = f"{c}__sp"
        if c not in out.columns:
            out[c] = pd.Series([""]*len(out), dtype="string")
        mask = blank(out[c])
        out.loc[mask, c] = out.loc[mask, spc]
        out.drop(columns=[spc], inplace=True, errors="ignore")

    # ------- diagnostics (after)
    print("AFTER coverage:")
    for c in all_targets:
        if c in out.columns:
            print(f"  {c:>35}: {pct(out[c]):6.1f}%")

    out.to_csv(args.out, index=False)
    print(f"[merge_spiders_into_enriched] wrote {args.out} rows={len(out)}")
    
    # Summary by context
    fundraising_coverage = sum(pct(out[c]) for c in targets_num + targets_txt if c.startswith("fundraising_") and c in out.columns)
    voting_coverage = sum(pct(out[c]) for c in targets_num + targets_txt if c.startswith("voting_") and c in out.columns)
    
    fundraising_count = len([c for c in targets_num + targets_txt if c.startswith("fundraising_") and c in out.columns])
    voting_count = len([c for c in targets_num + targets_txt if c.startswith("voting_") and c in out.columns])
    
    print(f"[merge_spiders_into_enriched] Final context coverage:")
    print(f"  Fundraising: {fundraising_coverage/fundraising_count:.1f}% average across {fundraising_count} columns")
    print(f"  Voting:      {voting_coverage/voting_count:.1f}% average across {voting_count} columns")

if __name__ == "__main__":
    main()
