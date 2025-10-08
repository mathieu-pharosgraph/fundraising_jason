#!/usr/bin/env python3
"""
s7h_featuredot_precompute.py — Build compact "winners per CBG" and party timeline
from feature-dot scores (no 12 segments).

Inputs
------
--featuredot   data/affinity/topic_affinity_by_cbg_featuredot.parquet
--enriched     data/affinity/reports/topics_enriched.csv   (optional; for topic timeline)
--outdir       data/affinity/compact

Outputs
-------
best_per_cbg_Dem.parquet   (period, cbg_id, best_label, best_score, second_label, second_score, margin)
best_per_cbg_GOP.parquet
topic_timeline.parquet     (period, std_topic, party, affinity_sum)
"""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

def _norm_key(s:str)->str:
    return re.sub(r"[^a-z0-9]+","", str(s).lower())

def _topic_map_from_enriched(enriched_csv: str) -> pd.DataFrame:
    if not enriched_csv or not Path(enriched_csv).exists():
        return pd.DataFrame(columns=["label_key","standardized_topic_names"])
    df = pd.read_csv(enriched_csv, dtype=str, low_memory=False)
    keep = [c for c in ["story_label","label","standardized_topic_names"] if c in df.columns]
    if not keep: 
        return pd.DataFrame(columns=["label_key","standardized_topic_names"])
    d = df[keep].copy()
    labcol = "story_label" if "story_label" in d.columns else "label"
    d["label_key"] = d[labcol].astype(str).map(_norm_key)
    d["standardized_topic_names"] = d["standardized_topic_names"].astype(str).fillna("")
    return d[["label_key","standardized_topic_names"]].dropna(subset=["label_key"]).drop_duplicates("label_key")

def _explode_std_topics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["standardized_topic_names"] = d["standardized_topic_names"].astype(str).fillna("")
    d = d.assign(std_topic=d["standardized_topic_names"].str.split(r"\s*;\s*|,\s*|\|\s*"))
    d = d.explode("std_topic", ignore_index=True)
    d["std_topic"] = d["std_topic"].astype(str).str.strip()
    d = d[d["std_topic"]!=""]
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featuredot", default="data/affinity/topic_affinity_by_cbg_featuredot.parquet")
    ap.add_argument("--enriched",   default="data/affinity/reports/topics_enriched.csv")
    ap.add_argument("--outdir",     default="data/affinity/compact")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load feature-dot dataset
    fd = ds.dataset(args.featuredot, format="parquet")
    cols = fd.schema.names

    # choose party columns (prefer blended)
    dem_col = "final_affinity_cbg_dem" if "final_affinity_cbg_dem" in cols else ("affinity_cbg_dem" if "affinity_cbg_dem" in cols else None)
    gop_col = "final_affinity_cbg_gop" if "final_affinity_cbg_gop" in cols else ("affinity_cbg_gop" if "affinity_cbg_gop" in cols else None)
    if not dem_col or not gop_col:
        raise SystemExit("Party columns not found in feature-dot file. Expected final_affinity_cbg_dem/gop or affinity_cbg_dem/gop.")

    # gather periods
    periods = sorted(set(pd.Series(fd.to_table(columns=["period"]).column("period").to_pylist(), dtype="string").dropna().astype(str)))
    if not periods:
        raise SystemExit("No period column/values found in feature-dot file.")

    def _winners_for_party(party: str, score_col: str) -> pd.DataFrame:
        rows = []
        for p in periods:
            tbl = fd.to_table(columns=["period","cbg_id","label",score_col], filter=(ds.field("period")==p))
            df = tbl.to_pandas()
            if df.empty: 
                continue
            df["__r"] = df.groupby("cbg_id")[score_col].rank(ascending=False, method="first")
            best = df[df["__r"]==1].copy()
            best = best.rename(columns={"label":"best_label", score_col:"best_score"})
            sec = (df[df.groupby("cbg_id")[score_col].rank(ascending=False, method="first")==2]
                   [["cbg_id","label",score_col]].rename(columns={"label":"second_label",score_col:"second_score"}))
            best = best.merge(sec, on="cbg_id", how="left")
            best["period"] = str(p)
            best["margin"] = (best["best_score"] - best["second_score"]).astype(float)
            rows.append(best[["period","cbg_id","best_label","best_score","second_label","second_score","margin"]])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["period","cbg_id","best_label","best_score","second_label","second_score","margin"])

    # winners
    dem = _winners_for_party("Dem", dem_col)
    gop = _winners_for_party("GOP", gop_col)
    dem.to_parquet(outdir/"best_per_cbg_Dem.parquet", index=False)
    gop.to_parquet(outdir/"best_per_cbg_GOP.parquet", index=False)

    # timeline: sum of party affinity by standardized topic and period
    # need label_key->std_topic mapping from enriched (optional)
    m = _topic_map_from_enriched(args.enriched)
    timeline_rows = []
    for party, col in [("Dem", dem_col), ("GOP", gop_col)]:
        for p in periods:
            tbl = fd.to_table(columns=["period","label",col], filter=(ds.field("period")==p))
            df = tbl.to_pandas()
            if df.empty: 
                continue
            df["label_key"] = df["label"].astype(str).map(_norm_key)
            if not m.empty:
                d2 = df.merge(m, on="label_key", how="left")
                d2 = _explode_std_topics(d2)
                d2["aff_sum"] = pd.to_numeric(d2[col], errors="coerce").fillna(0.0)
                g = (d2.groupby(["period","std_topic"], as_index=False)["aff_sum"].sum()
                       .assign(party=party)
                       .rename(columns={"aff_sum":"affinity_sum"}))
            else:
                # fallback: use raw labels as topics
                df["std_topic"] = df["label"].astype(str)
                df["affinity_sum"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                g = df.groupby(["period","std_topic"], as_index=False)["affinity_sum"].sum()
                g["party"] = party
            timeline_rows.append(g[["period","std_topic","party","affinity_sum"]])

    if timeline_rows:
        tl = pd.concat(timeline_rows, ignore_index=True)
    else:
        tl = pd.DataFrame(columns=["period","std_topic","party","affinity_sum"])
    tl.to_parquet(outdir/"topic_timeline.parquet", index=False)

    print("✓ wrote", outdir/"best_per_cbg_Dem.parquet", len(dem))
    print("✓ wrote", outdir/"best_per_cbg_GOP.parquet", len(gop))
    print("✓ wrote", outdir/"topic_timeline.parquet", len(tl))

if __name__ == "__main__":
    main()
