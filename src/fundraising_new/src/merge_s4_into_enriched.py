#!/usr/bin/env python3
import argparse, re
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path as _P
# add <repo_root>/src so "fundraising_new" is importable
sys.path.append(str(_P(__file__).resolve().parents[3] / "src"))
from fundraising_new.src.utils.keys import nkey



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)   # CSV: data/affinity/reports/topics_enriched.csv (BOTH CONTEXTS)
    ap.add_argument("--s4",       required=True)   # PARQUET: data/topics/political_classification_enriched.parquet (BOTH CONTEXTS)
    ap.add_argument("--s5",       default="data/topics/merged_data_with_topics.parquet")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--allow-any-period-fallback", action="store_true",
                    help="If a row still misses, fill from S4 by label_key ignoring period (use latest processed_at).")
    args = ap.parse_args()

    print(f"[merge_s4_into_enriched] Starting merge with both fundraising and voting contexts...")

    # ---------- load enriched (BOTH FUNDRAISING + VOTING) ----------
# ---------- load enriched (BOTH FUNDRAISING + VOTING) ----------
    en = pd.read_csv(args.enriched, low_memory=False)
    print(f"[merge_s4_into_enriched] Loaded enriched: {len(en)} rows, columns: {list(en.columns)}")

    en["period_norm"] = pd.to_datetime(en.get("period",""), errors="coerce").dt.date.astype("string")
    lab_en = "story_label" if "story_label" in en.columns else ("label" if "label" in en.columns else None)
    en["label_key"] = en[lab_en].astype(str).apply(nkey) if lab_en else ""
    if "cluster_id" in en.columns:
        en["cluster_id"] = pd.to_numeric(en["cluster_id"], errors="coerce").astype("Int64")
    else:
        en["cluster_id"] = pd.Series([pd.NA]*len(en), dtype="Int64")


    # ---------- backfill cluster_id from s5 (period_norm + label_key) ----------
    s5 = pd.read_parquet(args.s5)
    s5["period_norm"] = pd.to_datetime(s5.get("period",""), errors="coerce").dt.date.astype("string")
    s5["label_key"]   = s5.get("label", s5.get("story_label","")).astype(str).apply(nkey)
    m5 = s5[["period_norm","label_key","cluster_id"]].dropna().drop_duplicates(["period_norm","label_key"])
    before = en["cluster_id"].notna().sum()
    en = en.merge(m5.rename(columns={"cluster_id":"cluster_id__s5"}), on=["period_norm","label_key"], how="left")
    en["cluster_id"] = en["cluster_id"].where(en["cluster_id"].notna(),
                                              pd.to_numeric(en["cluster_id__s5"], errors="coerce").astype("Int64"))
    en.drop(columns=["cluster_id__s5"], inplace=True)
    after = en["cluster_id"].notna().sum()
    print(f"[merge_s4_into_enriched] Cluster ID backfill: {before} -> {after} non-null")

    # ---------- load S4 and normalize (BOTH FUNDRAISING + VOTING) ----------
    s4 = pd.read_parquet(args.s4)
    print(f"[merge_s4_into_enriched] Loaded S4: {len(s4)} rows, columns: {list(s4.columns)}")
    
    lab_s4 = "story_label" if "story_label" in s4.columns else ("label" if "label" in s4.columns else None)
    s4["period_norm"] = pd.to_datetime(s4.get("period",""), errors="coerce").dt.date.astype("string")
    s4["label_key"]   = s4[lab_s4].astype(str).apply(nkey) if lab_s4 else ""
    if "cluster_id" in s4.columns:
        s4["cluster_id"] = pd.to_numeric(s4["cluster_id"], errors="coerce").astype("Int64")

    # Check if S4 has context column (new format with both fundraising + voting)
    has_context = "context" in s4.columns
    print(f"[merge_s4_into_enriched] S4 has context column: {has_context}")
    
    if has_context:
        # NEW FORMAT: Split S4 by context and create proper column names
        print(f"[merge_s4_into_enriched] Processing S4 with context separation...")

        s4_fundraising = s4[s4["context"] == "fundraising"].copy()
        s4_voting      = s4[s4["context"] == "voting"].copy()

        print(f"[merge_s4_into_enriched] Found {len(s4_fundraising)} fundraising rows, {len(s4_voting)} voting rows")

        # Rename generic potentials to context-specific names
        s4_fundraising = s4_fundraising.rename(columns={
            "dem_potential": "dem_fundraising_potential",
            "gop_potential": "gop_fundraising_potential",
            "classification": "fundraising_classification",
            "dem_angle":      "fundraising_dem_angle",
            "gop_angle":      "fundraising_gop_angle",
        })
        s4_voting = s4_voting.rename(columns={
            "dem_potential": "dem_voting_potential",
            "gop_potential": "gop_voting_potential",
            "classification": "voting_classification",
            "dem_angle":      "voting_dem_angle",
            "gop_angle":      "voting_gop_angle",
        })

        # define final target columns per context
        fundraising_cols = [
            "fundraising_classification", "fundraising_dem_angle", "fundraising_gop_angle",
            "dem_fundraising_potential",  "gop_fundraising_potential",
        ]
        voting_cols = [
            "voting_classification", "voting_dem_angle", "voting_gop_angle",
            "dem_voting_potential",  "gop_voting_potential",
        ]

        keep_f = ["period_norm", "cluster_id", "label_key", "processed_at"] + fundraising_cols
        keep_v = ["period_norm", "cluster_id", "label_key", "processed_at"] + voting_cols

        s4_fundraising = s4_fundraising[[c for c in keep_f if c in s4_fundraising.columns]].copy()
        s4_voting      = s4_voting[[c for c in keep_v if c in s4_voting.columns]].copy()

    else:
        # OLD FORMAT: Assume direct columns exist; prefix classification/angles to avoid collisions
        print(f"[merge_s4_into_enriched] Processing S4 with direct columns...")

        # Build prefixed text columns
        f_text_map = {}
        v_text_map = {}
        if "classification" in s4.columns: f_text_map["classification"] = "fundraising_classification"
        if "dem_angle" in s4.columns:      f_text_map["dem_angle"]      = "fundraising_dem_angle"
        if "gop_angle" in s4.columns:      f_text_map["gop_angle"]      = "fundraising_gop_angle"
        if "classification" in s4.columns: v_text_map["classification"] = "voting_classification"
        if "dem_angle" in s4.columns:      v_text_map["dem_angle"]      = "voting_dem_angle"
        if "gop_angle" in s4.columns:      v_text_map["gop_angle"]      = "voting_gop_angle"

        s4_fundraising = s4.rename(columns=f_text_map).copy()
        s4_voting      = s4.rename(columns=v_text_map).copy()

        fundraising_cols = [c for c in [
            "fundraising_classification","fundraising_dem_angle","fundraising_gop_angle",
            "dem_fundraising_potential","gop_fundraising_potential"
        ] if c in s4_fundraising.columns]

        voting_cols = [c for c in [
            "voting_classification","voting_dem_angle","voting_gop_angle",
            "dem_voting_potential","gop_voting_potential"
        ] if c in s4_voting.columns]

        s4_fundraising = s4_fundraising[["period_norm","cluster_id","label_key"] + fundraising_cols].copy()
        s4_voting      = s4_voting[["period_norm","cluster_id","label_key"] + voting_cols].copy()


    print(f"[merge_s4_into_enriched] Fundraising columns to merge: {fundraising_cols}")
    print(f"[merge_s4_into_enriched] Voting columns to merge: {voting_cols}")

    # Keep the latest per (period_norm,cluster_id) and per (period_norm,label_key)
    # Keep the latest per key using processed_at, if present
    if "processed_at" in s4_fundraising.columns:
        s4_fundraising = s4_fundraising.sort_values("processed_at")
    if "processed_at" in s4_voting.columns:
        s4_voting = s4_voting.sort_values("processed_at")


    # Create primary and secondary keys for both contexts
    s4_fundraising_pk = s4_fundraising.drop_duplicates(["period_norm","cluster_id"], keep="last")
    s4_fundraising_lk = s4_fundraising.drop_duplicates(["period_norm","label_key"], keep="last")
    
    s4_voting_pk = s4_voting.drop_duplicates(["period_norm","cluster_id"], keep="last")
    s4_voting_lk = s4_voting.drop_duplicates(["period_norm","label_key"], keep="last")

    # Start with our enriched base (already has both contexts)
    out = en.copy()

    # ---------- MERGE FUNDRAISING DATA ----------
    print(f"[merge_s4_into_enriched] Merging fundraising data...")
    out = out.merge(s4_fundraising_pk[["period_norm","cluster_id"] + fundraising_cols]
                   .rename(columns={c:f"{c}__pri" for c in fundraising_cols}),
                   on=["period_norm","cluster_id"], how="left")

    out = out.merge(s4_fundraising_lk[["period_norm","label_key"] + fundraising_cols]
                    .rename(columns={c:f"{c}__sec" for c in fundraising_cols}),
                    on=["period_norm","label_key"], how="left")

    # ---------- MERGE VOTING DATA ----------
    print(f"[merge_s4_into_enriched] Merging voting data...")
    out = out.merge(s4_voting_pk[["period_norm","cluster_id"] + voting_cols]
                   .rename(columns={c:f"{c}__pri" for c in voting_cols}),
                   on=["period_norm","cluster_id"], how="left")

    out = out.merge(s4_voting_lk[["period_norm","label_key"] + voting_cols]
                    .rename(columns={c:f"{c}__sec" for c in voting_cols}),
                    on=["period_norm","label_key"], how="left")

    # ---------- tertiary (optional): label_key only, any period → latest processed_at ----------
    if args.allow_any_period_fallback:
        print(f"[merge_s4_into_enriched] Applying any-period fallback...")
        s4_fundraising_any = s4_fundraising.drop_duplicates(["label_key"], keep="last")
        s4_voting_any = s4_voting.drop_duplicates(["label_key"], keep="last")
        
        out = out.merge(s4_fundraising_any[["label_key"] + fundraising_cols]
                        .rename(columns={c:f"{c}__any" for c in fundraising_cols}),
                        on=["label_key"], how="left")
        out = out.merge(s4_voting_any[["label_key"] + voting_cols]
                        .rename(columns={c:f"{c}__any" for c in voting_cols}),
                        on=["label_key"], how="left")
    else:
        for c in fundraising_cols + voting_cols:
            out[f"{c}__any"] = pd.NA

    # ---------- fill priority: existing -> pri -> sec -> any ----------
    # Define numeric vs text columns
    num_cols = {
        "dem_fundraising_potential","gop_fundraising_potential",
        "dem_voting_potential","gop_voting_potential"
    }
    text_cols = {
        "fundraising_classification","fundraising_dem_angle","fundraising_gop_angle",
        "voting_classification","voting_dem_angle","voting_gop_angle",
    }
    all_cols = list(num_cols | text_cols)

    # optional any-period fallback stays the same but uses the new names
    if args.allow_any_period_fallback:
        print(f"[merge_s4_into_enriched] Applying any-period fallback...")
        s4_fundraising_any = s4_fundraising.drop_duplicates(["label_key"], keep="last")
        s4_voting_any      = s4_voting.drop_duplicates(["label_key"], keep="last")
        out = out.merge(
            s4_fundraising_any[["label_key"] + fundraising_cols]
              .rename(columns={c: f"{c}__any" for c in fundraising_cols}),
            on=["label_key"], how="left"
        )
        out = out.merge(
            s4_voting_any[["label_key"] + voting_cols]
              .rename(columns={c: f"{c}__any" for c in voting_cols}),
            on=["label_key"], how="left"
        )
    else:
        for c in all_cols:
            out[f"{c}__any"] = pd.NA

    # unified fill
    for c in all_cols:
        pri_c, sec_c, any_c = f"{c}__pri", f"{c}__sec", f"{c}__any"
        if c in num_cols:
            if c not in out.columns: out[c] = pd.NA
            out[c] = pd.to_numeric(out[c], errors="coerce")
            for src in (pri_c, sec_c, any_c):
                if src in out.columns:
                    out[src] = pd.to_numeric(out[src], errors="coerce")
                    out[c] = out[c].where(out[c].notna(), out[src])
                    out.drop(columns=[src], inplace=True)
        else:  # text
            if c not in out.columns: out[c] = ""
            for src in (pri_c, sec_c, any_c):
                if src in out.columns:
                    m_blank = out[c].astype(str).str.strip().isin(["", "nan", "None", "—", "-"]) | out[c].isna()
                    out.loc[m_blank, c] = out.loc[m_blank, src]
                    out.drop(columns=[src], inplace=True)


    # ---------- write ----------
    out.to_csv(args.out, index=False)
    print(f"[merge_s4_into_enriched] wrote {args.out}")
    print(f"[merge_s4_into_enriched] Final coverage statistics:")
    # ---------- write ----------
    out.to_csv(args.out, index=False)
    print(f"[merge_s4_into_enriched] wrote {args.out}")

    # ---------- (optional) coverage report ----------
    print(f"[merge_s4_into_enriched] Final coverage statistics:")

    def _pct(col):
        if col not in out.columns:
            return "MISSING"
        return f"{out[col].notna().mean():.1%}"

    print("  --- FUNDRAISING ---")
    for c in fundraising_cols:
        print(f"  {c}: {_pct(c)}")

    print("  --- VOTING ---")
    for c in voting_cols:
        print(f"  {c}: {_pct(c)}")



if __name__ == "__main__":
    main()
