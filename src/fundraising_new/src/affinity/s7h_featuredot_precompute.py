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

def _political_label_keys(s4_path: str, min_potential: int) -> set[str]:
    p = Path(s4_path)
    if not p.exists(): return set()
    s4 = pd.read_parquet(p)
    # normalize
    lab = None
    for c in ("story_label","label","winner_label","best_label"):
        if c in s4.columns: lab = c; break
    if lab is None: return set()

    s4["label_key"] = s4[lab].astype(str).str.lower().str.replace(r"[^a-z0-9]+","", regex=True)

    # exclude explicit non-political, and (optionally) require some fundraising potential
    s4 = s4[~s4["classification"].fillna("non-political").str.contains("non-political", case=False)]
    if min_potential > 0:
        for c in ("dem_fundraising_potential","gop_fundraising_potential"):
            if c in s4.columns:
                s4[c] = pd.to_numeric(s4[c], errors="coerce")
        s4 = s4[
            (s4.get("dem_fundraising_potential", 0).fillna(0).ge(min_potential)) |
            (s4.get("gop_fundraising_potential", 0).fillna(0).ge(min_potential))
        ]
    return set(s4["label_key"].dropna().unique().tolist())


def _winners(fd, score_col: str, periods: list[str], pol_keys: set[str]) -> pd.DataFrame:
    rows = []
    for p in periods:
        tbl = fd.to_table(columns=["period","cbg_id","label",score_col], filter=(ds.field("period")==p))
        df = tbl.to_pandas()
        if df.empty: continue
        if pol_keys:
            lk = df["label"].astype(str).str.lower().str.replace(r"[^a-z0-9]+","", regex=True)
            df = df[lk.isin(pol_keys)].copy()
            if df.empty: 
                continue
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
    ap.add_argument("--s4", default="data/topics/political_classification_enriched.parquet")
    ap.add_argument("--min_potential", type=int, default=0,
                help="Keep labels where max(dem,gop) >= this (0-100); 0 disables the threshold")
    args = ap.parse_args()

    # label_key -> standardized_topic_names, produced in step 2
    MAP_PATH = args.enriched  # pass the new file path when running
    m = pd.read_csv(MAP_PATH, dtype=str, low_memory=False)[["label_key","standardized_topic_names"]].drop_duplicates("label_key")


    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pol_keys = _political_label_keys(args.s4, args.min_potential)
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

    # winners by track (write only those present) — FILTER ALL BY pol_keys
    if dem_col:  _winners(fd, dem_col,  periods, pol_keys).to_parquet(outdir/"best_per_cbg_Dem.parquet", index=False)
    if gop_col:  _winners(fd, gop_col,  periods, pol_keys).to_parquet(outdir/"best_per_cbg_GOP.parquet", index=False)
    if any_col:  _winners(fd, any_col,  periods, pol_keys).to_parquet(outdir/"best_per_cbg_Any.parquet", index=False)
    if cand_col: _winners(fd, cand_col, periods, pol_keys).to_parquet(outdir/"best_per_cbg_Candidate.parquet", index=False)
    if org_col:  _winners(fd, org_col,  periods, pol_keys).to_parquet(outdir/"best_per_cbg_Org.parquet", index=False)
    if size_col: _winners(fd, size_col, periods, pol_keys).to_parquet(outdir/"best_per_cbg_Size.parquet", index=False)

    # ---- build timeline for whatever tracks exist ----
    tl_rows = []

    def _explode_and_sum(df_labels: pd.DataFrame, col: str, party: str | None, track: str, m: pd.DataFrame) -> pd.DataFrame | None:
        """
        df_labels: ['period','label','label_key', <col>]
        col      : affinity column to aggregate
        party    : 'Dem'|'GOP' or None
        track    : 'party'|'any'|'cand'|'org'|'size'
        m        : ['label_key','standardized_topic_names']
        """
        if df_labels.empty:
            return None

        d2 = df_labels.copy()

        # Merge standardized topics on label_key, fallback to raw label text
        if m is not None and not m.empty:
            d2 = d2.merge(m, on="label_key", how="left")
            d2["std_topic"] = d2["standardized_topic_names"].where(
                d2["standardized_topic_names"].notna(),
                d2["label"].astype(str)
            )
        else:
            d2["std_topic"] = d2["label"].astype(str)

        # Explode to single topics and sanitize
        d2 = d2.assign(topic=d2["std_topic"].astype(str)
                                .str.split(r"\s*;\s*|\s*\|\s*|,\s*")).explode("topic", ignore_index=True)
        d2["topic"] = d2["topic"].astype(str).str.strip()
        d2 = d2[(d2["topic"] != "") & (d2["topic"].str.lower() != "nan")]

        # Numeric coercion + aggregation
        d2["affinity_sum"] = pd.to_numeric(d2[col], errors="coerce").fillna(0.0)
        g = (d2.groupby(["period","topic"], as_index=False)["affinity_sum"].sum()
            .assign(track=track))

        if party is not None:
            g["party"] = party

        return g[["period","topic","track","party","affinity_sum"]] if "party" in g.columns else g[["period","topic","track","affinity_sum"]]

    tracks = []  # (track_name, party_or_None, column_name)
    if dem_col:  tracks.append(("party", "Dem", dem_col))
    if gop_col:  tracks.append(("party", "GOP", gop_col))
    if any_col:  tracks.append(("any",   None, any_col))
    if cand_col: tracks.append(("cand",  None, cand_col))
    if org_col:  tracks.append(("org",   None, org_col))
    if size_col: tracks.append(("size",  None, size_col))

    for track, party, col in tracks:
        for p in periods:
            have_lk = "label_key" in fd.schema.names
            cols_to_read = ["period","label","label_key",col] if have_lk else ["period","label",col]
            tbl = fd.to_table(columns=[c for c in cols_to_read if c in fd.schema.names],
                            filter=(ds.field("period")==p))
            df = tbl.to_pandas()
            if df.empty:
                continue

            if "label_key" not in df.columns:
                df["label_key"] = df["label"].astype(str).map(_norm_key)

            # political filter (use label_key — consistent with S4)
            if pol_keys:
                df = df[df["label_key"].isin(pol_keys)].copy()
                if df.empty:
                    continue

            g = _explode_and_sum(df, col, party, track, m)
            if g is not None and not g.empty:
                tl_rows.append(g)


    tl_cols = ["period","topic","track","party","affinity_sum"]
    tl = pd.concat(tl_rows, ignore_index=True)[tl_cols] if tl_rows else pd.DataFrame(columns=tl_cols)
    tl = tl.rename(columns={"topic":"std_topic"})
    tl.to_parquet(outdir/"topic_timeline.parquet", index=False)
    print("✓ precompute complete (timeline rows:", len(tl), ")")

if __name__ == "__main__":
    main()
