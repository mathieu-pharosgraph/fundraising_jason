#!/usr/bin/env python3
import argparse, re
import pandas as pd
from pathlib import Path

def nkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)   # CSV: data/affinity/reports/topics_enriched.csv
    ap.add_argument("--s4",       required=True)   # PARQUET: data/topics/political_classification_enriched.parquet
    ap.add_argument("--s5",       default="data/topics/merged_data_with_topics.parquet")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--allow-any-period-fallback", action="store_true",
                    help="If a row still misses, fill from S4 by label_key ignoring period (use latest processed_at).")
    args = ap.parse_args()

    # ---------- load enriched ----------
    en = pd.read_csv(args.enriched)
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

    # ---------- load S4 and normalize ----------
    s4 = pd.read_parquet(args.s4)
    lab_s4 = "story_label" if "story_label" in s4.columns else ("label" if "label" in s4.columns else None)
    s4["period_norm"] = pd.to_datetime(s4.get("period",""), errors="coerce").dt.date.astype("string")
    s4["label_key"]   = s4[lab_s4].astype(str).apply(nkey) if lab_s4 else ""
    if "cluster_id" in s4.columns:
        s4["cluster_id"] = pd.to_numeric(s4["cluster_id"], errors="coerce").astype("Int64")

    want = [c for c in ["classification","dem_angle","gop_angle",
                        "dem_fundraising_potential","gop_fundraising_potential"] if c in s4.columns]

    # keep the latest per (period_norm,cluster_id) and per (period_norm,label_key)
    if "processed_at" in s4.columns:
        s4 = s4.sort_values("processed_at")

    s4_pk = s4.drop_duplicates(["period_norm","cluster_id"], keep="last")
    s4_lk = s4.drop_duplicates(["period_norm","label_key"], keep="last")

    # ---------- primary merge: (period_norm, cluster_id) ----------
    out = en.merge(s4_pk[["period_norm","cluster_id"] + want]
                   .rename(columns={c:f"{c}__pri" for c in want}),
                   on=["period_norm","cluster_id"], how="left")

    # ---------- secondary merge: (period_norm, label_key) ----------
    out = out.merge(s4_lk[["period_norm","label_key"] + want]
                    .rename(columns={c:f"{c}__sec" for c in want}),
                    on=["period_norm","label_key"], how="left")

    # ---------- tertiary (optional): label_key only, any period → latest processed_at ----------
    if args.allow_any_period_fallback:
        s4_any = s4.drop_duplicates(["label_key"], keep="last")
        out = out.merge(s4_any[["label_key"] + want]
                        .rename(columns={c:f"{c}__any" for c in want}),
                        on=["label_key"], how="left")
    else:
        for c in want:
            out[f"{c}__any"] = pd.NA

    # ---------- fill priority: existing -> pri -> sec -> any ----------
    for c in want:
        pri_c, sec_c, any_c = f"{c}__pri", f"{c}__sec", f"{c}__any"
        if pri_c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[pri_c] = pd.to_numeric(out[pri_c], errors="coerce")
            out[c] = out[c].where(out[c].notna(), out[pri_c])
            out.drop(columns=[pri_c], inplace=True)
        if sec_c in out.columns:
            out[sec_c] = pd.to_numeric(out[sec_c], errors="coerce")
            out[c] = out[c].where(out[c].notna(), out[sec_c])
            out.drop(columns=[sec_c], inplace=True)
        if any_c in out.columns:
            out[any_c] = pd.to_numeric(out[any_c], errors="coerce")
            out[c] = out[c].where(out[c].notna(), out[any_c])
            out.drop(columns=[any_c], inplace=True)

    # ---------- write ----------
    out.to_csv(args.out, index=False)
    print("[merge_s4_into_enriched] wrote", args.out)
    for c in want:
        print(f"  coverage {c}: {out[c].notna().mean():.3%}")

if __name__ == "__main__":
    main()
