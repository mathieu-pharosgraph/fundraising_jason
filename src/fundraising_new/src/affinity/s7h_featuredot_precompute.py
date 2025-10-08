#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd, numpy as np
import pyarrow.dataset as ds

def _norm_key(s:str)->str: return re.sub(r"[^a-z0-9]+","", str(s).lower())
def _topic_map(enriched_csv: str) -> pd.DataFrame:
    if not enriched_csv or not Path(enriched_csv).exists(): return pd.DataFrame()
    df = pd.read_csv(enriched_csv, dtype=str, low_memory=False)
    lab = "story_label" if "story_label" in df.columns else ("label" if "label" in df.columns else None)
    if not lab or "standardized_topic_names" not in df.columns: return pd.DataFrame()
    df = df[[lab,"standardized_topic_names"]].copy()
    df["label_key"] = df[lab].astype(str).map(_norm_key)
    return df[["label_key","standardized_topic_names"]].drop_duplicates("label_key")

def _explode_topics(d: pd.DataFrame) -> pd.DataFrame:
    if "standardized_topic_names" not in d.columns: return pd.DataFrame()
    x = d.assign(std_topic=d["standardized_topic_names"].astype(str).str.split(r"\s*;\s*|,\s*|\|\s*"))
    x = x.explode("std_topic", ignore_index=True)
    x["std_topic"] = x["std_topic"].astype(str).str.strip()
    return x[x["std_topic"]!=""]

def _winners(fd, score_col: str, periods: list[str]) -> pd.DataFrame:
    rows = []
    for p in periods:
        tbl = fd.to_table(columns=["period","cbg_id","label",score_col], filter=(ds.field("period")==p))
        df = tbl.to_pandas()
        if df.empty: continue
        df["__r"] = df.groupby("cbg_id")[score_col].rank(ascending=False, method="first")
        best = df[df["__r"]==1].copy()
        best = best.rename(columns={"label":"best_label", score_col:"best_score"})
        sec  = (df[df.groupby("cbg_id")[score_col].rank(ascending=False, method="first")==2]
                [["cbg_id","label",score_col]].rename(columns={"label":"second_label",score_col:"second_score"}))
        best = best.merge(sec, on="cbg_id", how="left")
        best["period"] = str(p)
        best["margin"] = (best["best_score"] - best["second_score"]).astype(float)
        rows.append(best[["period","cbg_id","best_label","best_score","second_label","second_score","margin"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["period","cbg_id","best_label","best_score","second_label","second_score","margin"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featuredot", default="data/affinity/topic_affinity_by_cbg_featuredot.parquet")
    ap.add_argument("--enriched",   default="data/affinity/reports/topics_enriched.csv")
    ap.add_argument("--outdir",     default="data/affinity/compact")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fd = ds.dataset(args.featuredot, format="parquet")
    cols = fd.schema.names
    periods = sorted(set(pd.Series(fd.to_table(columns=["period"]).column("period").to_pylist(), dtype="string").dropna().astype(str)))
    if not periods: raise SystemExit("No period values in feature-dot file.")

    # discover columns for tracks
    choose = lambda *opts: next((c for c in opts if c in cols), None)
    dem_col  = choose("final_affinity_cbg_dem","affinity_cbg_dem")
    gop_col  = choose("final_affinity_cbg_gop","affinity_cbg_gop")
    any_col  = choose("affinity_cbg_any")           # will exist if you extend s7c2 to compute it
    cand_col = choose("affinity_cbg_cand")
    org_col  = choose("affinity_cbg_org")
    size_col = choose("affinity_cbg_size")
    if not dem_col or not gop_col:
        raise SystemExit("Need party columns (final_affinity_cbg_dem/gop or affinity_cbg_dem/gop).")

    # winners by track (write only those present)
    if dem_col:  _winners(fd, dem_col, periods).to_parquet(outdir/"best_per_cbg_Dem.parquet", index=False)
    if gop_col:  _winners(fd, gop_col, periods).to_parquet(outdir/"best_per_cbg_GOP.parquet", index=False)
    if any_col:  _winners(fd, any_col, periods).to_parquet(outdir/"best_per_cbg_Any.parquet", index=False)
    if cand_col: _winners(fd, cand_col, periods).to_parquet(outdir/"best_per_cbg_Candidate.parquet", index=False)
    if org_col:  _winners(fd, org_col, periods).to_parquet(outdir/"best_per_cbg_Org.parquet", index=False)
    if size_col: _winners(fd, size_col, periods).to_parquet(outdir/"best_per_cbg_Size.parquet", index=False)

    # timeline with track marker (here: party only, consistent with app Tab 1)
    m = _topic_map(args.enriched)
    tl_rows = []
    for party, col in [("Dem", dem_col), ("GOP", gop_col)]:
        for p in periods:
            tbl = fd.to_table(columns=["period","label",col], filter=(ds.field("period")==p))
            df = tbl.to_pandas()
            if df.empty: continue
            df["label_key"] = df["label"].astype(str).map(_norm_key)
            if not m.empty:
                d2 = df.merge(m, on="label_key", how="left")
                d2 = _explode_topics(d2)
                d2["aff_sum"] = pd.to_numeric(d2[col], errors="coerce").fillna(0.0)
                g = (d2.groupby(["period","std_topic"], as_index=False)["aff_sum"].sum()
                       .assign(party=party, track="party")
                       .rename(columns={"aff_sum":"affinity_sum"}))
            else:
                df["std_topic"] = df["label"].astype(str)
                df["affinity_sum"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                g = df.groupby(["period","std_topic"], as_index=False)["affinity_sum"].sum()
                g["party"] = party; g["track"] = "party"
            tl_rows.append(g[["period","std_topic","party","track","affinity_sum"]])
    tl = pd.concat(tl_rows, ignore_index=True) if tl_rows else pd.DataFrame(columns=["period","std_topic","party","track","affinity_sum"])
    tl.to_parquet(outdir/"topic_timeline.parquet", index=False)
    print("✓ precompute complete.")
if __name__ == "__main__":
    main()
