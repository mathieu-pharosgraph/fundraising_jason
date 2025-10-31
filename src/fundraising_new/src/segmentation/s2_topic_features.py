#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[3]))

from fundraising_new.src.utils.keys import nkey

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
    ap.add_argument("--context", choices=["both", "fundraising", "voting"], default="both",
                   help="Which context to build features for")
    args = ap.parse_args()

    df = pd.read_parquet(args.metrics).copy()
    df["period"] = df["period"].astype(str)
    
    # Base columns that are common to both contexts
    base_columns = ["period", "cluster_id", "label"]
    
    # Context-specific columns to process
    context_configs = []
    
    if args.context in ["both", "fundraising"]:
        fundraising_cols = {
            "urgency_score": "fundraising_urgency_score",
            "emotions": "fundraising_emotions", 
            "moral_foundations": "fundraising_moral_foundations",
            "hooks": "fundraising_hooks",
            "cta": "fundraising_cta",
            "emotions_top": "fundraising_emotions_top"
        }
        # Check if fundraising columns exist
        if all(col in df.columns for col in fundraising_cols.values()):
            context_configs.append(("fundraising", fundraising_cols))
        else:
            print("⚠️  Fundraising columns missing - skipping fundraising context")
    
    if args.context in ["both", "voting"]:
        voting_cols = {
            "urgency_score": "voting_urgency_score",
            "emotions": "voting_emotions",
            "moral_foundations": "voting_moral_foundations", 
            "hooks": "voting_hooks",
            "cta": "voting_cta",
            "emotions_top": "voting_emotions_top"
        }
        # Check if voting columns exist
        if all(col in df.columns for col in voting_cols.values()):
            context_configs.append(("voting", voting_cols))
        else:
            print("⚠️  Voting columns missing - skipping voting context")
    
    if not context_configs:
        raise ValueError("No valid context columns found to process")

    # Start with base columns
    X = df[base_columns].copy()
    X["label_key"] = X["label"].astype(str).map(nkey)

    # Process each context
    for context_name, col_map in context_configs:
        # Select columns for this context
        context_base = df[base_columns + list(col_map.values())].copy()
        
        # Rename columns to standard names for processing
        rename_map = {v: k for k, v in col_map.items()}
        context_base = context_base.rename(columns=rename_map)
        
        # Flatten nested JSON structures with context-specific prefixes
        prefix = f"{context_name}_"
        
        em = pd.json_normalize(context_base["emotions"].apply(to_dict)).add_prefix(f"{prefix}emo_")
        mf = pd.json_normalize(context_base["moral_foundations"].apply(to_dict)).add_prefix(f"{prefix}mf_")
        hooks = pd.json_normalize(context_base["hooks"].apply(to_dict)).add_prefix(f"{prefix}hook_")
        cta = pd.json_normalize(context_base["cta"].apply(to_dict)).add_prefix(f"{prefix}cta_")
        
        # Add urgency score and emotions_top
        context_flat = pd.concat([
            context_base[["period", "cluster_id", "label", "urgency_score", "emotions_top"]].rename(
                columns={
                    "urgency_score": f"{prefix}urgency_score",
                    "emotions_top": f"{prefix}emotions_top"
                }
            ),
            em, mf, hooks, cta
        ], axis=1)
        
        # Merge with main dataframe
        X = X.merge(context_flat, on=base_columns, how="left")

    # events / items per day (shared between contexts)
    if Path(args.events).exists():
        ev = pd.read_parquet(args.events).rename(columns={"day": "period"}).copy()

        # 1) normalize the count column name to 'items'
        count_col = None
        for cand in ("items", "count", "n_items", "n", "stories", "value"):
            if cand in ev.columns:
                count_col = cand
                break
        if count_col is None:
            # no usable count column; treat as empty
            ev["items"] = pd.NA
        elif count_col != "items":
            ev = ev.rename(columns={count_col: "items"})

        # 2) build label_key on the events side
        ev["label_key"] = ev["label"].astype(str).map(nkey)

        # 3) primary join on (period, label_key)
        X = X.merge(
            ev[["period", "label_key", "items"]],
            on=["period", "label_key"], how="left"
        )

        # make sure 'items' exists before fallback
        if "items" not in X.columns:
            X["items"] = pd.NA

        # 4) fallback: legacy join on (period, label) for older files
        miss = X["items"].isna()
        if miss.any() and {"period", "label", "items"} <= set(ev.columns):
            fb = (
                X.loc[miss, ["period", "label"]]
                .merge(ev[["period", "label", "items"]], on=["period", "label"], how="left")
            )
            X.loc[miss, "items"] = fb["items"].values

        # 5) finalize
        X = X.rename(columns={"items": "items_per_day"})
        X["items_per_day"] = pd.to_numeric(X["items_per_day"], errors="coerce").fillna(0).astype(int)
    else:
        X["items_per_day"] = 0



    # party lean debug (if present) - shared between contexts
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
    print("✓ Contexts processed:", [name for name, _ in context_configs])

if __name__ == "__main__":
    main()
