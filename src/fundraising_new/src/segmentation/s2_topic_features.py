#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd
from pathlib import Path


METRICS = Path("data/topics/metrics/topic_metrics.parquet")
EVENTS  = Path("data/topics/topic_events.parquet")
LEANDBG = Path("data/topics/_summary/party_lean_debug.csv")
OUT     = Path("data/topics/features/topic_features_daily.parquet")

def to_dict(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return {}
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(METRICS))
    ap.add_argument("--events",  default=str(EVENTS))
    ap.add_argument("--lean-debug", default=str(LEANDBG))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    df = pd.read_parquet(args.metrics).copy()
    df["period"] = df["period"].astype(str)
    base = df[["period","cluster_id","label","urgency_score","emotions","moral_foundations","fundraising_hooks","cta","emotions_top"]].copy()

    # numeric flatten
    em = pd.json_normalize(base["emotions"].apply(to_dict)).add_prefix("emo_")
    mf = pd.json_normalize(base["moral_foundations"].apply(to_dict)).add_prefix("mf_")
    hooks = pd.json_normalize(base["fundraising_hooks"].apply(to_dict)).add_prefix("hook_")
    cta = pd.json_normalize(base["cta"].apply(to_dict)).add_prefix("cta_")

    X = pd.concat([base.drop(columns=["emotions","moral_foundations","fundraising_hooks","cta"]),
                   em, mf, hooks, cta], axis=1)

    # events / items per day
    if Path(args.events).exists():
        ev = pd.read_parquet(args.events).rename(columns={"day":"period"})
        X = X.merge(ev, on=["period","label"], how="left").rename(columns={"items":"items_per_day"})
    else:
        X["items_per_day"]=0

    # party lean debug (if present)
    # ---- party lean from political_classification only (no CSV) ----


    cls_path = Path("data/topics/political_classification.parquet")
    if not cls_path.exists():
        print("[info] political_classification.parquet missing → default to Neutral")
        X["party_lean_final"] = "Neutral"
        X["party_score"] = 0.0
        X["party_confidence"] = 0.0
        X["partisan_edge"] = 0.0
    else:
        cls = pd.read_parquet(cls_path).copy()

        # minimal required columns
        req = ["cluster_id", "classification"]
        miss = [c for c in req if c not in cls.columns]
        if miss:
            raise ValueError(f"political_classification.parquet missing columns: {miss}")

        # normalize cluster_id
        cls["cluster_id"] = pd.to_numeric(cls["cluster_id"], errors="coerce")

        def map_lean(s: str) -> str:
            s = (s or "").strip().lower()
            if s == "dem-owned":   return "Dem"
            if s == "gop-owned":   return "GOP"
            if s == "contested":   return "Contested"   # keep as BOTH (not Neutral)
            if s == "dem-avoided": return "GOP"
            if s == "gop-avoided": return "Dem"
            return "Neutral"

        cls["party_lean_final"] = cls["classification"].apply(map_lean)

        # decide if raw potentials are available
        have_pots = {"dem_fundraising_potential","gop_fundraising_potential"}.issubset(cls.columns)

        merge_cols = ["cluster_id", "party_lean_final"]

        if have_pots:
            # coerce but DO NOT fabricate; keep NaNs if present
            for c in ("dem_fundraising_potential","gop_fundraising_potential"):
                cls[c] = pd.to_numeric(cls[c], errors="coerce")
            # derived fields only when real potentials exist
            cls["partisan_edge"] = cls["dem_fundraising_potential"] - cls["gop_fundraising_potential"]
            cls["party_score"]   = cls["partisan_edge"].abs()
            total = (cls["dem_fundraising_potential"] + cls["gop_fundraising_potential"]).replace(0, np.nan)
            cls["contest_index"] = (1 - (cls["partisan_edge"].abs() / total)).clip(0,1).fillna(0)

            cls["party_confidence"] = (
                (cls["dem_fundraising_potential"] + cls["gop_fundraising_potential"]) / 200.0
            ).clip(0.0, 1.0)

            merge_cols += [
                "party_score","party_confidence","partisan_edge",
                "dem_fundraising_potential","gop_fundraising_potential"
            ]

        # merge only what exists
        X = X.merge(cls[merge_cols], on="cluster_id", how="left")

        # Fill any non-classified clusters (e.g., below threshold) as Neutral — only if present
        if "party_lean_final" in X.columns:
            X["party_lean_final"] = X["party_lean_final"].fillna("Neutral")

        # Coerce derived numeric columns only if they exist (conditional merge may omit them)
        for c in ("party_score", "party_confidence", "partisan_edge"):
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)



    # clean
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(args.out, index=False)
    print("✓ wrote", args.out, "rows=", len(X))

if __name__ == "__main__":
    main()
