#!/usr/bin/env python3
import argparse, json, re, os, time
import numpy as np, pandas as pd
from pathlib import Path

# You already have deepseek_chat elsewhere; reuse if you want.
import requests as rq

def deepseek_chat(messages, model="deepseek-chat", max_tokens=600, temperature=0.1, timeout=60):
    key  = os.getenv("DEEPSEEK_API_KEY")
    base = os.getenv("DEEPSEEK_API_URL","https://api.deepseek.com/v1").rstrip("/")
    url  = f"{base}/chat/completions"
    if not key: raise SystemExit("Set DEEPSEEK_API_KEY")
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    pl  = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    r = rq.post(url, headers=hdr, json=pl, timeout=timeout); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

PROMPT = """You will score how a political STORY activates a compact set of donor-predictive FEATURES.
Return STRICT JSON ONLY like:
{{"weights": {{"feature_key": 0.0}}, "top": [{{"feature_key":"...", "why":"..."}}]}}

Rules:
- Only use features from this list (exact keys): {feature_keys}
- Weights in [-1, 1], 0 = not relevant. Be sparse (<= 8 non-zeros).
- Positive weight = story increases that feature's fundraising relevance; negative = decreases.
- Provide 2-3 short "why" justifications in the "top" array for the strongest weights.

STORY (period={period}, label={label}):
{snippet}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics-dir", default="data/topics")
    ap.add_argument("--basis", default="data/donors/feature_basis.parquet")
    ap.add_argument("--out", default="data/topics/story_feature_affinity.parquet")
    ap.add_argument("--max-weights", type=int, default=8)
    ap.add_argument("--model", default="deepseek-chat")
    args = ap.parse_args()

    # features we allow the scorer to use
    B = pd.read_parquet(args.basis)
    all_keys = sorted(B["feature_key"].astype(str).unique().tolist())

    # Optional: restrict to keys that actually exist (or can be aliased) in CBG features
    # This keeps the LLM’s menu aligned with what the feature-dot stage can use later.
    try:
        C = pd.read_parquet("fundraising_participation/data/geo/cbg_features.parquet")
        df_cols = set(C.columns)

        # minimal alias map (mirror of your trainer/mapper)
        alias = {
            "income": ["median_hh_income","median_income","B19013_001E"],
            "educ": ["pct_bachelors_plus","pct_bachelor_plus","pct_ba_plus"],
            "internet_home": ["pct_broadband","broadband_rate"],
            "home_owner": ["owner_occ_rate"],
            "urban_simple": ["urban_q","metro_micro","cbsa_cat_num"],
            "poverty_rate": ["poverty_rate"],
            "median_gross_rent": ["median_gross_rent"],
            "median_home_value": ["median_home_value"],
            "rent_as_income_pct": ["rent_as_income_pct"],
            "ntee_public_affairs_per_1k": ["ntee_public_affairs_per_1k"],
            "ntee_total_per_1k": ["ntee_total_per_1k"],
            "share_dem": ["share_dem"], "share_gop": ["share_gop"],
            "emp_share_manufacturing": ["emp_share_manufacturing"],
            "emp_share_retail": ["emp_share_retail"],
            "emp_share_healthcare_social": ["emp_share_healthcare_social"],
            "emp_share_professional_scientific_mgmt": ["emp_share_professional_scientific_mgmt"],
            "emp_share_information": ["emp_share_information"],
            # optional:
            "emp_share_agriculture": ["emp_share_agriculture"],
            "emp_share_other_services": ["emp_share_other_services"],
        }

        def mappable(k):
            if k in df_cols: 
                return True
            for cand in alias.get(k, []):
                if cand in df_cols:
                    return True
            return False

        feature_keys = [k for k in all_keys if mappable(k)]
        # if for some reason we filtered everything, fall back to all_keys
        if not feature_keys:
            feature_keys = all_keys
    except Exception:
        feature_keys = all_keys

    # Representative text (reuse your S3 items/clusters)
    items   = pd.read_parquet(Path(args.topics_dir)/"items.parquet")
    clusters= pd.read_parquet(Path(args.topics_dir)/"clusters.parquet")
    meta    = pd.read_parquet(Path(args.topics_dir)/"cluster_meta.parquet")

    # --- NEW: keep only accepted political/fundraising clusters ---
    try:
        S4 = pd.read_parquet("data/topics/political_classification_enriched.parquet")
        # accept anything that isn’t explicitly non-political; optional: require ≥ some potential
        S4 = S4[~S4["classification"].fillna("non-political").str.contains("non-political", case=False)]
        # optional potency gate (uncomment if you want)
        # S4 = S4[(S4["dem_fundraising_potential"] >= 30) | (S4["gop_fundraising_potential"] >= 30)]

        keep_ids = set(pd.to_numeric(S4["cluster_id"], errors="coerce").dropna().astype(int))
        meta = meta[meta["cluster_id"].isin(keep_ids)].copy()
        if meta.empty:
            print("⚠️ meta restricted to political/fundraising=EMPTY — check S4")
    except Exception as e:
        print(f"⚠️ could not apply political filter via S4: {e}")


    items = items[["item_id","title","source","url","published_at","text"]]
    df = (items.merge(clusters[["item_id","cluster_id","cluster_prob"]], on="item_id", how="inner")
               .merge(meta[["cluster_id","label"]], on="cluster_id", how="left"))
    df["period"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.date.astype(str)

    rows=[]
    for (period,label), sub in df.groupby(["period","label"]):
        sub = sub.sort_values(["cluster_prob","published_at"], ascending=[False,False]).head(10)
        snippet = ""
        for _, r in sub.iterrows():
            t = (r.get("title") or "")[:100]
            s = (r.get("source") or "")
            x = (r.get("text") or "")[:300]
            u = (r.get("url") or "")
            snippet += f"- {t} — {s}\n  {x}\n  {u}\n"
        messages = [
            {"role":"system","content":"You are a careful scoring assistant. JSON only."},
            {"role":"user", "content": PROMPT.format(feature_keys=", ".join(feature_keys),
                                                     period=period, label=label, snippet=snippet)}
        ]
        try:
            raw = deepseek_chat(messages, model=args.model, max_tokens=700, temperature=0.1)
            s = raw.strip().strip("`").strip()
            j = json.loads(re.search(r"\{.*\}", s, re.S).group(0))
            w = j.get("weights", {}) or {}
            # sparsify to max-weights
            if len(w) > args.max_weights:
                # keep top by absolute value
                w = dict(sorted(w.items(), key=lambda kv: abs(float(kv[1]) if kv[1] is not None else 0.0),
                                reverse=True)[:args.max_weights])
            for k,v in w.items():
                try: rows.append({"period":period,"label":label,
                                  "feature_key":str(k),"weight":float(v)})
                except: continue
        except Exception as e:
            # skip on failure
            continue
        time.sleep(0.15)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"✓ wrote {args.out} rows={len(out)}")

if __name__ == "__main__":
    main()
