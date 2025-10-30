#!/usr/bin/env python3
"""
Enhanced political classification with hashing for cache management.
Now supports BOTH fundraising and voting contexts.
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import hashlib
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
import pyarrow.dataset as ds

# Load environment variables
from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parents[2] / "secret.env"
load_dotenv(ENV_PATH)

# DeepSeek API setup
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")

MAP_DIR = "data/topics/merged_data_with_topics.parquet"

# ================= DUAL CONTEXT PROMPTS =================

FUNDRAISING_CLASSIFICATION_PROMPT = """
You are a political analyst specializing in US politics and fundraising. Today is {current_date}.
Donald Trump is the CURRENT President of the United States, leading a Republican administration.
Republicans control the White House and executive branch (including the DOJ).

Topic label: "{topic_label}"
Standardized categories: {standardized_topics}

For EACH party, produce a concise fundraising angle that would motivate **donors** of that party,
and numeric fundraising potentials 0–100 (integers).

Guidance
- GOP angles: attack Democratic positions, defend Republican actions, highlight threats from Democrats/deep state/external foes.
  Since GOP controls the DOJ, avoid alleging corruption **by** GOP-controlled DOJ unless framed as deep-state sabotage.
- DEM angles: attack GOP actions/abuse/overreach; defend norms/rights; highlight threats to democracy/corruption under Trump admin.
- Keep each angle ≤160 chars. Be realistic to **today's** context.

Return ONLY strict JSON with this exact schema (no extra keys, no text):
{{
  "story_label": "<short human label>",
  "classification": "dem-owned" | "gop-owned" | "contested" | "dem-avoided" | "gop-avoided" | "non-political",
  "dem_angle": "<string>",
  "gop_angle": "<string>",
  "dem_fundraising_potential": <0-100>,
  "gop_fundraising_potential": <0-100>
}}

Cluster items:
{items_text}
"""

VOTING_CLASSIFICATION_PROMPT = """
You are a political analyst specializing in US politics and voter mobilization. Today is {current_date}.
Donald Trump is the CURRENT President of the United States, leading a Republican administration.
Republicans control the White House and executive branch (including the DOJ).

Topic label: "{topic_label}"
Standardized categories: {standardized_topics}

For EACH party, produce a concise voter mobilization angle that would motivate **voters** of that party,
and numeric voting potentials 0–100 (integers).

Guidance
- GOP angles: emphasize conservative values, border security, economic growth, anti-woke themes, election integrity.
- DEM angles: emphasize protection of democracy, abortion rights, climate change, social justice, voting rights.
- Focus on issues that drive voter turnout and engagement, not donations.
- Keep each angle ≤160 chars. Be realistic to **today's** context.

Return ONLY strict JSON with this exact schema (no extra keys, no text):
{{
  "story_label": "<short human label>",
  "classification": "dem-owned" | "gop-owned" | "contested" | "dem-avoided" | "gop-avoided" | "non-political",
  "dem_angle": "<string>",
  "gop_angle": "<string>",
  "dem_voting_potential": <0-100>,
  "gop_voting_potential": <0-100>
}}

Cluster items:
{items_text}
"""

def nkey(s): return re.sub(r"[^a-z0-9]+","", str(s).lower())

def deepseek_chat(messages: List[Dict[str, str]], model="deepseek-chat",
                  max_tokens=300, temperature=0.1):
    """Direct implementation of deepseek_chat function"""
    base = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
    url = f"{base.rstrip('/')}/chat/completions"
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in env")
    
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model, 
        "messages": messages,
        "max_tokens": max_tokens, 
        "temperature": temperature
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=(10, 30))
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def get_current_date_context():
    """Get the current date and format it for context priming"""
    now = pd.Timestamp.now()
    current_date = now.strftime("%B %d, %Y")
    
    # Calculate how long Trump has been in office (assuming Jan 20, 2025 inauguration)
    inauguration_date = pd.Timestamp("2025-01-20")
    days_in_office = (now - inauguration_date).days
    months_in_office = days_in_office // 30
    
    return f"{current_date} (Trump has been in office for {months_in_office} months since January 20, 2025)"

def _clip01(x):
    try:
        return float(max(0, min(100, float(x))))
    except Exception:
        return None

def safe_json_load(s: str) -> Dict[str, Any]:
    """Safely parse JSON from LLM response"""
    s = s.strip().strip("`").strip()
    try:
        return json.loads(s)
    except Exception:
        # Try to extract JSON from text if it's wrapped
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    # Return a default error response if parsing fails
    return {
        "classification": "error",
        "gop_angle": "",
        "dem_angle": "",
        "reasoning": "Failed to parse JSON response",
        "fundraising_potential": {"gop": 0, "dem": 0},
        "voting_potential": {"gop": 0, "dem": 0}
    }

def compute_content_hash(items_df: pd.DataFrame) -> str:
    """Compute a hash of the content to detect changes."""
    content_str = ""
    for _, row in items_df.iterrows():
        content_str += f"{row.get('title', '')}|{row.get('text', '')}|"
    return hashlib.sha256(content_str.encode()).hexdigest()

# Fixed retry decorator
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def classify_topic_politically(topic_label: str, items_text: str, standardized_topics: List[str], context: str = "fundraising") -> Dict[str, Any]:
    """Classify a topic's political positioning using DeepSeek for specified context."""
    topics_str = ", ".join(standardized_topics) if isinstance(standardized_topics, list) else str(standardized_topics or "")
    
    if context == "voting":
        prompt = VOTING_CLASSIFICATION_PROMPT.format(
            current_date=get_current_date_context(),
            topic_label=topic_label,
            items_text=items_text,
            standardized_topics=topics_str
        )
    else:
        prompt = FUNDRAISING_CLASSIFICATION_PROMPT.format(
            current_date=get_current_date_context(),
            topic_label=topic_label,
            items_text=items_text,
            standardized_topics=topics_str
        )

    messages = [
        {"role": "system", "content": f"You are a political analyst specializing in US politics and {context}."},
        {"role": "user", "content": prompt}
    ]
    response = deepseek_chat(messages, max_tokens=400, temperature=0.1)
    return safe_json_load(response)

def format_items_text(items_df: pd.DataFrame, max_items: int = 8) -> str:
    """Format representative items for the LLM prompt."""
    items_text = ""
    for i, (_, row) in enumerate(items_df.iterrows()):
        if i >= max_items:
            break
            
        title = row.get('title', '')[:100]  # Truncate title
        content = row.get('text', '')[:300]  # Truncate content
        source = row.get('source', 'unknown')
        
        items_text += f"Item {i+1} (Source: {source}):\n"
        items_text += f"Title: {title}\n"
        items_text += f"Content: {content}\n\n"
    
    return items_text

def load_topic_data(topics_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items    = pd.read_parquet(topics_dir / "items.parquet")
    clusters = pd.read_parquet(topics_dir / "clusters.parquet")
    meta     = pd.read_parquet(topics_dir / "cluster_meta.parquet")

    # Synthesize generic label/party_lean from fundraising→voting fallbacks
    f_lbl = meta.get("fundraising_label")
    v_lbl = meta.get("voting_label")
    if f_lbl is not None or v_lbl is not None:
        meta["label"] = (
            (f_lbl.astype(str) if f_lbl is not None else "")
            .where((f_lbl.notna() if f_lbl is not None else False) &
                (f_lbl.astype(str).str.strip() != ""), v_lbl)
            .fillna("")
            .astype(str)
        )

    f_pl = meta.get("fundraising_party_lean")
    v_pl = meta.get("voting_party_lean")
    if f_pl is not None or v_pl is not None:
        meta["party_lean"] = (
            (f_pl.astype(str) if f_pl is not None else "")
            .where((f_pl.notna() if f_pl is not None else False) &
                (f_pl.astype(str).str.strip() != ""), v_pl)
            .fillna("")
            .astype(str)
        )

    # Optionally enrich meta with standardized topics (if present)
    std_path = topics_dir / "merged_data_with_topics.parquet"
    if std_path.exists():
        std = pd.read_parquet(std_path)
        # keep only id/label + topic columns; and avoid blowing away meta rows
        keep = [c for c in ["cluster_id","label","standardized_topic_names","standardized_topic_ids"] if c in std.columns]
        if keep:
            meta = meta.merge(std[keep], on=["cluster_id","label"], how="left")
    return items, clusters, meta

def load_or_create_cache(cache_path: Path) -> pd.DataFrame:
    """Load existing cache or create a new one."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    else:
        return pd.DataFrame(columns=[
            "cluster_id", "content_hash", "context", "classification", "gop_angle", 
            "dem_angle", "gop_potential", "dem_potential", "processed_at"
        ])

def validate_angle(topic_label: str, angle: str, party: str, context: str) -> str:
    """Validate a political angle using LLM. Returns 'VALID' or error message."""
    current_context = get_current_date_context()
    
    if context == "voting":
        validation_prompt = f"""
Validate this political voter mobilization angle for {party}. Today is {current_context}:

TOPIC: {topic_label}
ANGLE: {angle}

Check for:
1. Focus on voter motivation and turnout, not donations.
2. Relevance to the topic.
3. Strategic soundness for voter mobilization (highlights stakes for voters).
4. Correct temporal context (Trump is the CURRENT president).

If valid, respond with "VALID". Otherwise, respond with "INVALID: [reason]".
"""
    else:
        validation_prompt = f"""
Validate this political fundraising angle for {party}. Today is {current_context}:

TOPIC: {topic_label}
ANGLE: {angle}

Check for:
1. Factual plausibility given current political reality (Republicans control White House/DOJ).
2. Relevance to the topic.
3. Strategic soundness for fundraising (highlights threat/opportunity for the party).
4. Absence of contradictions (e.g., GOP alleging DOJ corruption under Trump doesn't make sense).
5. Correct temporal context (Trump is the CURRENT president, not former president).

If valid, respond with "VALID". Otherwise, respond with "INVALID: [reason]".
"""
    messages = [
        {"role": "system", "content": "You are a political strategist. Be concise and strict."},
        {"role": "user", "content": validation_prompt}
    ]
    try:
        response = deepseek_chat(messages, max_tokens=150, temperature=0)
        if "VALID" in response:
            return "VALID"
        else:
            return response.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"
    
def is_empty_rep_item_ids(rep_item_ids) -> bool:
    """Safely check if rep_item_ids is empty."""
    if isinstance(rep_item_ids, (list, np.ndarray)):
        return len(rep_item_ids) == 0
    elif pd.isna(rep_item_ids):
        return True
    else:
        # Handle other cases (e.g., string representation of list)
        try:
            if isinstance(rep_item_ids, str):
                parsed = json.loads(rep_item_ids)
                return len(parsed) == 0
        except:
            return True
        return False

def main():
    parser = argparse.ArgumentParser(description="Add political classification to topic analysis")
    parser.add_argument("--topics-dir", default="data/topics", help="Directory with topic data")
    parser.add_argument("--output-dir", default="data/topics", help="Output directory")
    parser.add_argument("--fundraising-threshold", type=int, default=60, 
                       help="Minimum fundraising score to include (0-100)")
    parser.add_argument("--voting-threshold", type=int, default=60,
                       help="Minimum voting score to include (0-100)")
    parser.add_argument("--context", choices=["both", "fundraising", "voting"], default="both",
                       help="Which context to classify")
    parser.add_argument("--max-items", type=int, default=8,
                       help="Maximum number of representative items to use per cluster")
    parser.add_argument("--cache-file", default="political_classification_cache.parquet",
                       help="Cache file to avoid reprocessing")
    args = parser.parse_args()
    
    topics_dir = Path(args.topics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading topic data...")
    items, clusters, meta = load_topic_data(topics_dir)
    
    # Determine which contexts to process
    contexts_to_process = []
    if args.context in ["both", "fundraising"]:
        # Filter for fundraising-relevant clusters
        fundraising_meta = meta[
            (meta["fundraising_us_relevance"]) & 
            (meta["fundraising_usable"]) & 
            (meta["fundraising_score"] >= args.fundraising_threshold)
        ].copy()
        if not fundraising_meta.empty:
            contexts_to_process.append(("fundraising", fundraising_meta))
    
    if args.context in ["both", "voting"]:
        # Filter for voting-relevant clusters  
        voting_meta = meta[
            (meta["voting_us_relevance"]) & 
            (meta["voting_usable"]) & 
            (meta["voting_score"] >= args.voting_threshold)
        ].copy()
        if not voting_meta.empty:
            contexts_to_process.append(("voting", voting_meta))
    
    if not contexts_to_process:
        print("No relevant clusters found for the specified contexts and thresholds")
        return
    
    # Load cache
    cache_path = output_dir / args.cache_file
    cache_df = load_or_create_cache(cache_path)
    print(f"Cache contains {len(cache_df)} entries")
    
    # Process each context
    all_results = []
    for context_name, context_meta in contexts_to_process:
        print(f"\n=== Processing {context_name.upper()} context ===")
        print(f"Found {len(context_meta)} relevant clusters with {context_name} score >= {getattr(args, f'{context_name}_threshold')}")
        
        # Get clusters to process for this context
        clusters_to_process = []
        for _, row in context_meta.iterrows():
            cluster_id = row["cluster_id"]
            
            rep_item_ids = row.get("rep_item_ids", [])
            
            # Check if rep_item_ids is empty using our safe function
            if is_empty_rep_item_ids(rep_item_ids):
                continue
                
            # Convert to list if it's a string representation
            if isinstance(rep_item_ids, str):
                try:
                    rep_item_ids = json.loads(rep_item_ids)
                except:
                    continue
                    
            # Get the representative items
            rep_items = items[items["item_id"].isin(rep_item_ids)]
            if len(rep_items) == 0:
                continue
            
            # Compute content hash to check if we need to reprocess
            content_hash = compute_content_hash(rep_items)
            
            # Check if we have a cached result for this cluster with the same content and context
            cached_result = cache_df[
                (cache_df["cluster_id"] == cluster_id) & 
                (cache_df["content_hash"] == content_hash) &
                (cache_df["context"] == context_name)
            ]
            
            if not cached_result.empty:
                # Skip if we have a cached result for this content and context
                print(f"Skipping cluster {cluster_id} for {context_name} (already processed with same content)")
                continue

            # Get standardized topics
            standardized_topics = row.get('standardized_topic_names', [])
            if isinstance(standardized_topics, str):
                try:
                    standardized_topics = json.loads(standardized_topics)
                except:
                    standardized_topics = [standardized_topics]
            elif not isinstance(standardized_topics, list):
                standardized_topics = []
                
            clusters_to_process.append({
                "cluster_id": cluster_id,
                "label": row["label"],
                "rep_items": rep_items,
                "content_hash": content_hash,
                "standardized_topics": standardized_topics,
                "context": context_name
            })
        
        # Deduplicate
        seen_keys = set()
        deduped = []
        for _c in clusters_to_process:
            k = (int(_c["cluster_id"]), str(_c["content_hash"]), _c["context"])
            if k in seen_keys:
                continue
            seen_keys.add(k)
            deduped.append(_c)
        clusters_to_process = deduped
        
        print(f"Processing {len(clusters_to_process)} clusters for {context_name} political classification")
        
        # Process each cluster for this context
        context_results = []
        for cluster in clusters_to_process:
            cluster_id = cluster["cluster_id"]
            label = cluster["label"]
            rep_items = cluster["rep_items"]
            content_hash = cluster["content_hash"]
            standardized_topics = cluster["standardized_topics"]
            context_name = cluster["context"]
            
            print(f"Processing cluster {cluster_id} for {context_name}: {label}")
            
            # Format items for LLM
            items_text = format_items_text(rep_items, args.max_items)
            
            try:
                # Get political classification for this context
                classification_result = classify_topic_politically(
                    label, 
                    items_text, 
                    standardized_topics,
                    context_name
                )
                
                # Normalize schema
                if context_name == "fundraising":
                    potential_key = "fundraising_potential"
                    dem_potential = classification_result.get("dem_fundraising_potential")
                    gop_potential = classification_result.get("gop_fundraising_potential")
                else:
                    potential_key = "voting_potential" 
                    dem_potential = classification_result.get("dem_voting_potential")
                    gop_potential = classification_result.get("gop_voting_potential")
                
                # Extract angles
                gop_angle = classification_result.get("gop_angle", "")
                dem_angle = classification_result.get("dem_angle", "")

                # Validate angles
                gop_validation = validate_angle(label, gop_angle, "GOP", context_name)
                if "VALID" not in gop_validation:
                    print(f"GOP angle invalid for cluster {cluster_id} ({context_name}): {gop_validation}")
                
                dem_validation = validate_angle(label, dem_angle, "DEM", context_name)
                if "VALID" not in dem_validation:
                    print(f"DEM angle invalid for cluster {cluster_id} ({context_name}): {dem_validation}")

                # Create result
                result = {
                    "cluster_id": cluster_id,
                    "content_hash": content_hash,
                    "context": context_name,
                    "label": label,
                    "story_label": classification_result.get("story_label", label),
                    "classification": classification_result.get("classification", "unknown"),
                    "gop_angle": gop_angle,
                    "dem_angle": dem_angle,
                    "gop_potential": _clip01(gop_potential),
                    "dem_potential": _clip01(dem_potential),
                    "processed_at": pd.Timestamp.now()
                }
                
                context_results.append(result)
                
                # Update cache
                try:
                    cache_df = pd.concat([
                        cache_df,
                        pd.DataFrame([{
                            "cluster_id": int(cluster_id), 
                            "content_hash": content_hash,
                            "context": context_name
                        }])
                    ], ignore_index=True).drop_duplicates(["cluster_id", "content_hash", "context"])
                    cache_df.to_parquet(cache_path, index=False)
                except Exception:
                    pass
                
                print(f"  {context_name.capitalize()} Classification: {result['classification']}")
                
                # Be kind to the API
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing cluster {cluster_id} for {context_name}: {str(e)}")
                # Add error record
                context_results.append({
                    "cluster_id": cluster_id,
                    "content_hash": content_hash,
                    "context": context_name,
                    "label": label,
                    "classification": "error",
                    "error": str(e),
                    "processed_at": pd.Timestamp.now()
                })
        
        all_results.extend(context_results)
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Attach period via cluster_id from merged_data_with_topics.parquet
        mp = ds.dataset(MAP_DIR, format="parquet").to_table(columns=["cluster_id","period"]).to_pandas()
        mp["period"] = mp["period"].astype(str)
        mp = mp.drop_duplicates(["cluster_id","period"])

        results_df["story_label"] = results_df["label"].astype(str)
        results_df = results_df.merge(mp, on="cluster_id", how="left") 

        # Normalize numeric types and clamp to 0..100
        for c in ["dem_potential", "gop_potential"]:
            results_df[c] = pd.to_numeric(results_df.get(c, np.nan), errors="coerce").clip(0,100)

        # Create label_key for stable merges
        results_df["label_key"] = (
            results_df["story_label"]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "", regex=True)
            .str.strip()
        )
        results_df["period"] = results_df["period"].astype(str)
        results_df["cluster_id"] = pd.to_numeric(results_df["cluster_id"], errors="coerce").astype("Int64")
        
        # Save enriched results
        enriched_out = output_dir / "political_classification_enriched.parquet"
        results_df.to_parquet(enriched_out, index=False)
        print(f"Saved political classification (enriched) to {enriched_out}")

        # Update cache with new results
        cache_df = pd.concat([cache_df, results_df[["cluster_id", "content_hash", "context"]]], ignore_index=True)
        cache_df = cache_df.drop_duplicates(["cluster_id", "content_hash", "context"])
        cache_df.to_parquet(cache_path, index=False)
        print(f"Updated cache with {len(results_df)} new entries: {cache_path}")
        
        # Create comprehensive results file
        all_classifications_path = output_dir / "political_classification.parquet"
        results_df.to_parquet(all_classifications_path, index=False)
        print(f"Saved political classification to {all_classifications_path}")
        
        # Create separate files for each context
        for context_name in results_df["context"].unique():
            context_df = results_df[results_df["context"] == context_name].copy()
            context_path = output_dir / f"political_classification_{context_name}.parquet"
            context_df.to_parquet(context_path, index=False)
            print(f"Saved {context_name} political classification to {context_path}")
            
            # Create angles table for this context
            angles_df = context_df[["cluster_id", "label", "classification", "gop_angle", "dem_angle"]].copy()
            angles_path = output_dir / f"political_angles_{context_name}.parquet"
            angles_df.to_parquet(angles_path, index=False)
            print(f"Saved {context_name} political angles to {angles_path}")
        
    else:
        print("No new clusters processed")

if __name__ == "__main__":
    main()
