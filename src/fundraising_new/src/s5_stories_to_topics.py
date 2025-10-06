import pandas as pd
import requests
import time
import hashlib
import json
import os
from pathlib import Path

# 1. Load data using the same approach as your Streamlit app
def load_and_merge_data():
    """Load and merge data from the same files as the Streamlit app"""
    base_path = Path("data/topics")
    
    data_files = {
        "metrics": base_path / "metrics" / "topic_metrics.parquet",
        "political": base_path / "political_classification.parquet",
        "meta": base_path / "cluster_meta.parquet"
    }
    
    loaded_data = {}
    
    for name, path in data_files.items():
        try:
            if path.exists():
                loaded_data[name] = pd.read_parquet(path)
                print(f"Loaded {name} data: {len(loaded_data[name])} rows")
            else:
                print(f"File not found: {path}")
                loaded_data[name] = pd.DataFrame()
        except Exception as e:
            print(f"Error loading {name} data: {str(e)}")
            loaded_data[name] = pd.DataFrame()
    
    # Merge the data like your Streamlit app does
    metrics_df = loaded_data.get("metrics", pd.DataFrame())
    political_df = loaded_data.get("political", pd.DataFrame())
    meta_df = loaded_data.get("meta", pd.DataFrame())
    
    # Merge metrics with political classification
    merged_df = metrics_df
    if not political_df.empty:
        if 'cluster_id' in metrics_df.columns and 'cluster_id' in political_df.columns:
            merged_df = metrics_df.merge(
                political_df, 
                on='cluster_id', 
                how='left',
                suffixes=('', '_political')
            )
        elif 'label' in metrics_df.columns and 'label' in political_df.columns:
            merged_df = metrics_df.merge(
                political_df, 
                on='label', 
                how='left',
                suffixes=('', '_political')
            )
    
    # Merge with meta data if available
    if not meta_df.empty and not merged_df.empty and 'cluster_id' in merged_df.columns and 'cluster_id' in meta_df.columns:
        merged_df = merged_df.merge(
            meta_df[['cluster_id', 'us_relevance', 'fundraising_usable', 'fundraising_score']],
            on='cluster_id',
            how='left'
        )
    
    return merged_df

# Load the merged data
merged_df = load_and_merge_data()
print(f"Merged DataFrame has {len(merged_df)} rows")

# Get the unique labels from the merged data
cluster_labels = merged_df['label'].unique().tolist()
print(f"Found {len(cluster_labels)} unique labels to classify")

cache_file_path = "data/topics/classification_cache.json"

def load_cache():
    """Load the existing classification cache from a JSON file."""
    try:
        with open(cache_file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return an empty dict if file doesn't exist or is invalid
        return {}

def save_cache(cache):
    """Save the current classification cache to a JSON file."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    with open(cache_file_path, 'w') as f:
        json.dump(cache, f, indent=4)

def get_hash(input_string):
    """Generate a stable hash for a given input string."""
    return hashlib.sha256(input_string.strip().encode('utf-8')).hexdigest()

# 2. Configure the DeepSeek API Classifier
API_KEY = #modify to load the key via secret.env  # Replace with your actual key
API_URL = "https://api.deepseek.com/v1/chat/completions"  # Verify the correct endpoint

# Our expertly crafted system prompt
SYSTEM_PROMPT = """
You are a expert political and news content classifier. Your task is to assign one or more relevant topics from a defined taxonomy to a given news story headline.

TAXONOMY:
0. Non-Political / Other
1. Abortion & Reproductive Rights
2. Immigration Policy & Enforcement
3. Trump Administration & Policy Agenda
4. Election Integrity & Voting Rights
5. Healthcare Policy (ACA, Medicare, Medicaid)
6. Supreme Court & Judicial Affairs
7. Gun Policy & Violence
8. LGBTQ+ Rights
9. Economic Policy & indicators
10. Student Debt & Loan Forgiveness
11. Vaccines & Public Health
12. Climate Change & Environmental Policy
13. Foreign Policy & National Security
14. Ukraine-Russia War
15. Israel-Palestine Conflict
16. Civil Liberties & Free Speech
17. Law Enforcement & Policing
18. Congressional Dynamics & Legislation
19. Corporate Accountability & Business
20. Technology & AI Regulation
21. Labor & Workers' Rights
22. Education Policy
23. Social Security & Welfare Programs
24. Media & Journalism
25. Entertainment & Culture
26. Ethics & Corruption Scandals
27. Extremism & Domestic Threats
28. Censorship & Misinformation
29. Federal Agency Oversight
30. State vs. Federal Power
31. Refugee & Asylum Seeker Crisis
32. International Human Rights
33. Campaign Finance & Politics
34. Historical Legacy & Commemoration
35. Crime & Public Safety
36. Housing and Affordability
37. Opioid Crisis & Substance Abuse
38. Taxation & Fiscal Policy

INSTRUCTIONS:
- Analyze the user-provided headline.
- If the story is clearly NOT about politics, government, policy, or related societal issues, assign it ONLY to category 0 (Non-Political / Other).
- Select ALL topics from the taxonomy that are directly relevant to the story. A story can belong to 1, 2, 3, or occasionally more topics.
- Be precise. If a story is about a specific person acting (e.g., Trump), it should be tagged with their topic (3).
- If a story is about a state challenging federal law, tag it with the relevant policy topic AND (30).
- Output ONLY a comma-separated list of the exact topic numbers (e.g., "3, 30"). Do not include any other text, explanation, or formatting in your response.
"""

def classify_with_deepseek(headline, max_retries=3):
    """
    Sends a headline to the DeepSeek API for classification.
    Returns a list of topic numbers.
    """
    for attempt in range(max_retries):
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": headline}
            ],
            "temperature": 0.0,
            "max_tokens": 20
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            classification_text = response_data['choices'][0]['message']['content'].strip()
            
            # Check if the response is empty
            if not classification_text:
                print(f"Empty response for headline: {headline}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
                return []
                
            # Clean the response: remove periods, extra spaces, and split into a list of integers.
            topic_numbers = [int(num.strip()) for num in classification_text.split(',')]
            return topic_numbers

        except requests.exceptions.RequestException as e:
            print(f"API request failed for headline '{headline}' (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
                continue
            return []
        except (KeyError, ValueError) as e:
            print(f"Failed to parse response for headline '{headline}'. Response: '{classification_text}'. Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
                continue
            return []
    
    return []  # If all retries failed

# 3. Classify the Labels in Batch
# Load the existing cache
classification_cache = load_cache()
standardized_topics = []  # This will store the results for ALL labels, cached or new

# It's good practice to add a delay between API calls to avoid rate limits
delay_between_calls = 0.1  # 100 ms delay

for label in cluster_labels:
    label_hash = get_hash(label)
    
    # Check if we have a cached result for this hash
    if label_hash in classification_cache:
        # Use the cached result
        topic_list = classification_cache[label_hash]
        print(f"Cache hit for: {label}")
    else:
        # This is a new label, call the API
        topic_list = classify_with_deepseek(label)
        # Cache the result for future runs
        classification_cache[label_hash] = topic_list
        print(f"Classified and cached: {label} -> {topic_list}")
        time.sleep(delay_between_calls)  # Wait before making the next API call
    
    # Append the result (whether from cache or API) to our main list
    standardized_topics.append(topic_list)

# After processing all labels, save the updated cache
# This will include any new classifications from this run
save_cache(classification_cache)

# 4. Create a mapping from label to topic IDs and apply it to the merged DataFrame
label_to_topics = dict(zip(cluster_labels, standardized_topics))
merged_df['standardized_topic_ids'] = merged_df['label'].map(label_to_topics)

# Map the topic IDs to topic names
topic_id_to_name = {
    0: "Non-Political / Other",
    1: "Abortion & Reproductive Rights",
    2: "Immigration Policy & Enforcement",
    3: "Trump Administration & Policy Agenda",
    4: "Election Integrity & Voting Rights",
    5: "Healthcare Policy (ACA, Medicare, Medicaid)",
    6: "Supreme Court & Judicial Affairs",
    7: "Gun Policy & Violence",
    8: "LGBTQ+ Rights",
    9: "Economic Policy & indicators",
    10: "Student Debt & Loan Forgiveness",
    11: "Vaccines & Public Health",
    12: "Climate Change & Environmental Policy",
    13: "Foreign Policy & National Security",
    14: "Ukraine-Russia War",
    15: "Israel-Palestine Conflict",
    16: "Civil Liberties & Free Speech",
    17: "Law Enforcement & Policing",
    18: "Congressional Dynamics & Legislation",
    19: "Corporate Accountability & Business",
    20: "Technology & AI Regulation",
    21: "Labor & Workers' Rights",
    22: "Education Policy",
    23: "Social Security & Welfare Programs",
    24: "Media & Journalism",
    25: "Entertainment & Culture",
    26: "Ethics & Corruption Scandals",
    27: "Extremism & Domestic Threats",
    28: "Censorship & Misinformation",
    29: "Federal Agency Oversight",
    30: "State vs. Federal Power",
    31: "Refugee & Asylum Seeker Crisis",
    32: "International Human Rights",
    33: "Campaign Finance & Politics",
    34: "Historical Legacy & Commemoration",
    35: "Crime & Public Safety",
    36: "Housing and Affordability",
    37: "Opioid Crisis & Substance Abuse",
    38: "Taxation & Fiscal Policy"
}

merged_df['standardized_topic_names'] = merged_df['standardized_topic_ids'].apply(
    lambda id_list: [topic_id_to_name.get(tid, "Unknown") for tid in id_list] if isinstance(id_list, list) else []
)

# Save the enriched DataFrame back to a new Parquet file
output_path = "data/topics/merged_data_with_topics.parquet"
merged_df.to_parquet(output_path, index=False)

print(f"Classification complete! Results saved to: {output_path}")
