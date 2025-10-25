import os
import json
import datetime as dt
import time
import requests
import praw
#import snscrape.modules.twitter as sntwitter
from collections import defaultdict, Counter
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import random
import schedule  # You'll need to pip install schedule
# --- Dynamic expansion helpers ---
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from glob import glob
import numpy as np
from typing import Tuple
from datetime import datetime, timezone

# Calculate the path to the .env file
current_dir = Path(__file__).parent
env_path = current_dir.parent.parent / "secret.env"

# --- new env controls (top of file, after load_dotenv)
NEWSAPI_RPM = int(os.getenv("NEWSAPI_RPM", "120"))        # tune to your paid tier reality
REDDIT_RPM  = int(os.getenv("REDDIT_RPM", "50"))          # PRAW-friendly ceiling
NEWSAPI_MAX_PAGES = int(os.getenv("NEWSAPI_MAX_PAGES", "5"))
NEWSAPI_PAGE_SIZE = int(os.getenv("NEWSAPI_PAGE_SIZE", "100"))  # 100 is common max


# Add this after your existing imports
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist

# Load environment variables from the correct path
load_dotenv(dotenv_path=env_path)

# Configuration - Load from environment variables for security
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'PoliticalFundraisingScraper/1.0')

# ================= CONFIGURATION PARAMETERS =================
# Adjust these values to control the scale of data collection
CONFIG = {
    "max_topics_per_day": 60,          # Maximum topics to process daily
    "max_items_per_topic": 20,         # Maximum items to collect per topic
    "subreddits_to_monitor": ["politics", "Conservative", "PoliticalDiscussion"],
    "collection_windows": [            # Spread collection across the day
        {"start_hour": 2, "duration_hours": 4, "topics": 15},
        {"start_hour": 8, "duration_hours": 4, "topics": 15},
        {"start_hour": 14, "duration_hours": 4, "topics": 15},
        {"start_hour": 20, "duration_hours": 4, "topics": 15},
    ],
    "reddit_delay_between_requests": 2.5,  # Seconds between Reddit API calls
    "reddit_delay_between_topics": 8.0,    # Seconds between processing different topics
    "reddit_delay_between_subreddits": 12.0, # Seconds between subreddit checks
    "newsapi_delay": 1.5,                  # Seconds between NewsAPI calls
    "max_retries": 3,                      # Maximum retries for failed requests
    "enable_twitter": False,               # Twitter scraping is prone to blocking
}

# Seed topics - these will evolve over time
SEED_TOPICS = [
    # Core Political Issues (prioritized)
    "abortion", "gun control", "immigration", "climate change", "medicare",
    "reproductive rights", "roe v wade", "pro life", "pro choice",
    "second amendment", "gun violence", "NRA",
    "border security", "border wall", "dreamers", "daca",
    "green new deal", "environmental protection",
    "healthcare",  "affordable care act", "obamacare",
    "taxes", "tax cuts", "tax reform", "inflation", "economy",
    "election security", "voting rights", "voter suppression", "voter ID",
    
    # Additional topics (will be used if we have capacity)
    "critical race theory", "parental rights in education", "lgbtq rights", 
    "transgender rights", "student loans", "student debt forgiveness", 
    "social security", "supreme court", "scotus", "congress", "senate", 
    "house of representatives", "democracy", "freedom", "liberty", 
    "civil rights", "social justice", "woke", "socialism", "communism", 
    "fascism", "corruption", "infrastructure", "china", "russia", "ukraine", 
    "israel", "palestine"
]

TOPIC_TAXONOMY = {
  "abortion_repro_rights": {
    "include": ["abortion", "reproductive rights", "roe v wade", "dobbs decision", "planned parenthood", "heartbeat bill"],
    "exclude": ["veterinary", "animal breeding"]
  },
  "immigration_policy": {
    "include": ["immigration policy", "border security", "asylum seekers", "daca", "dreamers", "deportation", "ice", "cbp"],
    "exclude": []
  },
  "trump_policy_agenda": {
    "include": ["trump administration", "trump policy", "executive order", "maga agenda", "project 2025"],
    "exclude": []
  },
  "election_integrity": {
    "include": ["election integrity", "voter suppression", "voting rights", "voter id", "ballot access", "election security"],
    "exclude": []
  },
  "healthcare_policy": {
    "include": ["affordable care act", "obamacare", "medicare", "medicaid", "drug pricing", "prescription costs"],
    "exclude": []
  },
  "supreme_court": {
    "include": ["supreme court", "scotus", "certiorari", "oral arguments", "opinion issued"],
    "exclude": []
  },
  "gun_policy": {
    "include": ["gun control", "gun violence", "second amendment", "assault weapons ban", "nra"],
    "exclude": []
  },
  "lgbtq_rights": {
    "include": ["lgbtq rights", "transgender rights", "bathroom bill", "gender affirming care"],
    "exclude": []
  },
  "economy_macro": {
    "include": ["economic policy", "inflation", "jobs report", "cpi", "unemployment rate", "fed rate", "gdp"],
    "exclude": ["crypto price tips"]
  },
  "student_debt": {
    "include": ["student loans", "loan forgiveness", "borrower defense", "save plan"],
    "exclude": []
  },
  "public_health_vaccines": {
    "include": ["vaccines", "public health", "cdc guidance", "covid booster", "pandemic preparedness"],
    "exclude": []
  },
  "climate_env": {
    "include": ["climate change", "epa regulation", "renewable energy", "green new deal", "carbon emissions"],
    "exclude": []
  },
  "foreign_policy_natsec": {
    "include": ["foreign policy", "national security", "state department", "defense policy", "nato", "china policy"],
    "exclude": []
  },
  "ukraine_russia_war": {
    "include": ["ukraine war", "russia invasion", "aid to ukraine"],
    "exclude": []
  },
  "israel_palestine": {
    "include": ["israel palestine", "gaza", "idf", "ceasefire", "settlements"],
    "exclude": []
  },
  "civil_liberties_speech": {
    "include": ["free speech", "first amendment", "civil liberties", "aCLU litigation"],
    "exclude": ["campus sports speech awards"]  # example noise
  },
  "policing_public_safety": {
    "include": ["law enforcement", "policing reform", "qualified immunity", "public safety"],
    "exclude": []
  },
  "congress_legislation": {
    "include": ["congress", "house bill", "senate bill", "committee markup", "appropriations"],
    "exclude": []
  },
  "corporate_accountability": {
    "include": ["corporate accountability", "antitrust", "consumer protection", "ftc", "doj antitrust"],
    "exclude": []
  },
  "tech_ai_reg": {
    "include": ["ai regulation", "tech policy", "section 230", "privacy bill", "algorithms"],
    "exclude": []
  },
  "labor_workers_rights": {
    "include": ["labor rights", "unionization", "strike", "nlrb", "minimum wage", "overtime rule"],
    "exclude": []
  },
  "education_policy": {
    "include": ["education policy", "curriculum standards", "school board", "title ix (education)"],
    "exclude": ["ncaa title ix sports litigation"]  # optional
  },
  "social_security_welfare": {
    "include": ["social security", "welfare programs", "snap benefits", "ssi"],
    "exclude": []
  },
  "media_journalism": {
    "include": ["media bias", "press freedom", "journalism", "defamation case"],
    "exclude": []
  },
  "entertainment_culture": {
    "include": ["culture war", "hollywood strike", "awards controversy"],
    "exclude": []
  },
  "ethics_corruption": {
    "include": ["ethics investigation", "corruption scandal", "hatch act", "inspector general"],
    "exclude": []
  },
  "extremism_domestic": {
    "include": ["domestic extremism", "militia", "hate crime", "counterterrorism"],
    "exclude": []
  },
  "censorship_misinfo": {
    "include": ["censorship", "misinformation", "content moderation", "disinformation"],
    "exclude": []
  },
  "federal_oversight": {
    "include": ["agency oversight", "gao report", "oig report", "congressional oversight"],
    "exclude": []
  },
  "state_federal_power": {
    "include": ["states rights", "preemption", "federalism", "10th amendment"],
    "exclude": []
  },
  "refugee_asylum": {
    "include": ["refugee crisis", "asylum seekers", "resettlement program"],
    "exclude": []
  },
  "intl_human_rights": {
    "include": ["international human rights", "un human rights council", "icc"],
    "exclude": []
  },
  "campaign_finance_politics": {
    "include": ["campaign finance", "fec filing", "dark money", "super pac"],
    "exclude": []
  },
  "history_commemoration": {
    "include": ["historical legacy", "statue removal", "commemoration", "national archive"],
    "exclude": []
  },
  "crime_public_safety": {
    "include": ["crime rates", "public safety", "criminal justice reform", "bail reform"],
    "exclude": []
  },
  "housing_affordability": {
    "include": ["housing affordability", "zoning reform", "rent control", "mortgage rates"],
    "exclude": []
  },
  "opioids_substance_abuse": {
    "include": ["opioid crisis", "overdose deaths", "settlement", "harm reduction"],
    "exclude": []
  },
  "taxation_fiscal": {
    "include": ["tax policy", "tax cuts", "budget deficit", "debt ceiling", "irs enforcement"],
    "exclude": []
  }
}


# ================= RATE LIMITING SYSTEM =================
class RateLimiter:
    """Smart rate limiter to stay within API limits"""
    
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self.min_delay = 60.0 / requests_per_minute
    
    def wait_if_needed(self):
        """Wait if necessary to stay under rate limit"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            # Calculate when we can make the next request
            oldest_request = min(self.request_times)
            wait_time = 60 - (now - oldest_request) + 0.1  # Small buffer
            if wait_time > 0:
                time.sleep(wait_time)
        
        self.request_times.append(now)
        
        # Always maintain minimum delay
        time.sleep(self.min_delay)

# Initialize rate limiters
reddit_limiter = RateLimiter(REDDIT_RPM)
newsapi_limiter = RateLimiter(NEWSAPI_RPM)

# ================= CORE FUNCTIONS =================

def _daterange_chunks(start_dt: datetime, end_dt: datetime, days: int = 3):
    """Yield (chunk_start, chunk_end) in [start, end) with chunk size in days."""
    cur = start_dt
    one_day = dt.timedelta(days=1)
    step = dt.timedelta(days=max(1, days))
    while cur < end_dt:
        nxt = min(cur + step, end_dt)
        yield cur, nxt
        cur = nxt

def _window_days(start_dt: datetime | None, end_dt: datetime | None) -> int:
    if not start_dt or not end_dt:
        return 0
    return max(1, int((end_dt - start_dt).total_seconds() // 86400))


def _parse_date_utc(s: str | None) -> datetime | None:
    if not s: return None
    # Accept YYYY-MM-DD or full ISO
    try:
        if len(s) == 10:
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(s)
    except Exception:
        # last resort
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            raise SystemExit(f"Invalid date: {s} (use YYYY-MM-DD or ISO8601)")

def _within_window(ts_utc: float | int | None, start_dt: datetime | None, end_dt: datetime | None) -> bool:
    if ts_utc is None:
        return False
    t = datetime.fromtimestamp(float(ts_utc), tz=timezone.utc)
    if start_dt and t < start_dt:
        return False
    if end_dt and t >= end_dt:
        return False
    return True

def taxonomy_queries() -> List[Tuple[str, str]]:
    """
    Build (category, query) pairs from TOPIC_TAXONOMY using include terms
    plus NOT-excludes as simple guards.
    """
    pairs: List[Tuple[str, str]] = []
    for cat, rules in TOPIC_TAXONOMY.items():
        inc = [t.strip() for t in rules.get("include", []) if t.strip()]
        exc = [t.strip() for t in rules.get("exclude", []) if t.strip()]
        for term in inc:
            q = " ".join([f'"{term}"'] + [f'-"{x}"' for x in exc])
            pairs.append((cat, q))
    return pairs


def seed_queries(base_seeds: List[str]) -> List[Tuple[str, str]]:
    """
    Tag seeds under 'seed' category and return (category, query) pairs.
    """
    return [("seed", f'"{s}"') for s in base_seeds]


def label_categories_from_taxonomy(title: str, description: str, content: str) -> List[str]:
    """
    Post-hoc category labels using TOPIC_TAXONOMY include/exclude rules.
    """
    t = " ".join([(title or ""), (description or ""), (content or "")]).lower()
    out: List[str] = []
    for cat, rules in TOPIC_TAXONOMY.items():
        inc = [x.lower() for x in rules.get("include", [])]
        exc = [x.lower() for x in rules.get("exclude", [])]
        if inc and not any(x in t for x in inc):
            continue
        if exc and any(x in t for x in exc):
            continue
        out.append(cat)
    return out



def setup_reddit_client():
    """Initialize and return a Reddit client using PRAW"""
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        # Test the connection
        user = reddit.user.me()
        print(f"✓ Reddit client authenticated successfully as: {user}")
        return reddit
    except Exception as e:
        print(f"✗ Failed to initialize Reddit client: {str(e)}")
        return None

def get_newsapi_articles(
    query: str,
    page_size: int = None,
    max_pages: int = None,
    search_in="title,description",
    from_hours: int = 72,        # fallback if from_dt not provided
    language: str = "en",
    sort_by: str = "publishedAt",
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> List[Dict[str, Any]]:

    page_size = page_size or NEWSAPI_PAGE_SIZE
    max_pages = max_pages or NEWSAPI_MAX_PAGES

    articles = []
    base_url = "https://newsapi.org/v2/everything"

    def _iso_z(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    if from_dt:
        from_param = _iso_z(from_dt)
    else:
        now = dt.datetime.utcnow().replace(tzinfo=timezone.utc)
        from_param = (now - dt.timedelta(hours=from_hours)).isoformat(timespec="seconds").replace("+00:00", "Z")

    to_param = _iso_z(to_dt) if to_dt else None


    page = 1
    consecutive_empty = 0
    backoff = 2.0

    while page <= max_pages:
        newsapi_limiter.wait_if_needed()

        params = {
            "q": query,
            "searchIn": search_in,
            "sortBy": sort_by,
            "apiKey": NEWSAPI_KEY,
            "pageSize": page_size,
            "page": page,
            "language": language,
            "from": from_param,
        }
        if to_param:
            params["to"] = to_param


        try:
            resp = requests.get(base_url, params=params, timeout=20)
            if resp.status_code == 429:
                # rate limited: backoff and retry this same page
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            if resp.status_code >= 500:
                # transient server error: backoff and retry
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            if resp.status_code != 200:
                print(f"NewsAPI error {resp.status_code} for '{query}' p{page}: {resp.text[:200]}")
                break

            data = resp.json()
            if data.get("status") != "ok":
                print(f"NewsAPI non-ok for '{query}' p{page}: {data}")
                break

            page_articles = []
            for art in data.get("articles", []):
                content = art.get("content") or ""
                if content in (None, "[Removed]", "[+ chars]"):
                    content = art.get("description") or ""

                page_articles.append({
                    "source": "newsapi",
                    "title": art.get("title", ""),
                    "content": content,
                    "description": art.get("description", ""),
                    "url": art.get("url", ""),
                    "published_at": art.get("publishedAt", ""),
                    "source_name": art.get("source", {}).get("name", ""),
                    "author": art.get("author", ""),
                    "query": query
                })

            if not page_articles:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    # two empty pages in a row: likely exhausted
                    break
            else:
                consecutive_empty = 0
                articles.extend(page_articles)

            # If fewer than page_size came back, stop early
            if len(page_articles) < page_size:
                break

            page += 1
            backoff = 2.0  # reset on success

        except requests.RequestException as e:
            print(f"NewsAPI request exception for '{query}' p{page}: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            # retry same page

    return articles

def _dedup_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        url = (it.get("url") or "").strip().lower()
        if not url:
            # fallback key
            key = (it.get("source",""), it.get("source_name",""), it.get("title","").strip().lower(), str(it.get("published_at","")))
        else:
            key = ("url", url)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def get_reddit_posts(reddit_client, subreddit_name: str, topic: str, time_filter: str = 'day', limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch posts from a specific subreddit about a specific topic"""
    reddit_limiter.wait_if_needed()
    
    posts = []
    try:
        subreddit = reddit_client.subreddit(subreddit_name)
        
        # Search for the topic in this subreddit
        for post in subreddit.search(query=topic, sort='top', time_filter=time_filter, limit=limit):
            posts.append({
                'source': 'reddit',
                'subreddit': subreddit_name,
                'title': post.title,
                'content': post.selftext,
                'url': f"https://reddit.com{post.permalink}",
                'published_at': post.created_utc,
                'score': post.score,
                'num_comments': post.num_comments,
                'post_id': post.id,
                'query': topic
            })
            
    except Exception as e:
        print(f"Exception fetching Reddit posts from {subreddit_name} for {topic}: {str(e)}")
        # If we hit a rate limit, wait longer before continuing
        if "rate limit" in str(e).lower() or "too many" in str(e).lower():
            print("Rate limit detected. Waiting 120 seconds before continuing...")
            time.sleep(120)
    
    return posts

def get_twitter_posts(query: str, since_days: int = 1, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch tweets using snscrape with improved error handling"""
    if not CONFIG["enable_twitter"]:
        return []
        
    tweets = []
    since_date = (dt.datetime.now() - dt.timedelta(days=since_days)).strftime('%Y-%m-%d')
    
    try:
        search_query = f"{query} since:{since_date}"
        print(f"Attempting to scrape Twitter for: {search_query}")
        
        scraper = sntwitter.TwitterSearchScraper(search_query)
        
        for i, tweet in enumerate(scraper.get_items()):
            if i >= limit:
                break
                
            tweets.append({
                'source': 'twitter',
                'content': tweet.rawContent,
                'url': tweet.url,
                'published_at': tweet.date.timestamp(),
                'username': tweet.user.username,
                'retweet_count': tweet.retweetCount,
                'like_count': tweet.likeCount,
                'quote_count': tweet.quoteCount,
                'reply_count': tweet.replyCount,
                'query': query
            })
            
    except Exception as e:
        print(f"Twitter scraping failed for {query}: {str(e)}")
    
    return posts

def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- Robust TF-IDF keyphrase miner (handles tiny corpora safely) ---
def top_phrases(texts, topk=12, ngram_range=(1,3), min_df=2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = []
    for t in texts or []:
        s = clean_text(t)
        if s and len(s.split()) >= 3:
            docs.append(s)

    if not docs:
        return []

    n_docs = len(docs)

    # Adapt df thresholds to corpus size
    md = 1 if n_docs < 8 else min_df                  # min_df of 1 for very small corpora
    mx = 0.9                                          # default max_df (as proportion)
    if n_docs * mx < md:                              # avoid max_df < min_df
        mx = md / n_docs + 1e-6

    # Helper to filter trivial tokens
    bad = {"the","and","for","with","from","about","after","before","against","into","over",
           "under","this","that","there","their","them","they","a","an","to","of","in","on"}
    def _filter_vocab(vocab):
        out = []
        for term in vocab:
            toks = term.split()
            if term.isdigit(): 
                continue
            if all(tok in bad for tok in toks):
                continue
            out.append(term)
        return out

    # Try preferred settings; fall back progressively if needed
    for ngr in [(1,3), (1,2), (1,1)]:
        try:
            vec = TfidfVectorizer(ngram_range=ngr, min_df=md, max_df=mx, max_features=20000)
            X = vec.fit_transform(docs)
            vocab = list(vec.get_feature_names_out())
            if not vocab:
                continue
            scores = X.sum(axis=0).A1
            pairs = sorted(zip(vocab, scores), key=lambda z: z[1], reverse=True)
            kept = _filter_vocab([p[0] for p in pairs])
            if kept:
                return kept[:topk]
        except ValueError as e:
            # Catch 'max_df < min_df' and any other edge; relax settings
            md = 1
            mx = 1.0
            continue

    # Final fallback: simple unigram frequency
    from collections import Counter
    cnt = Counter()
    for s in docs:
        cnt.update([w for w in s.split() if w not in bad and len(w) > 2])
    return [w for w,_ in cnt.most_common(topk)]


def expand_from_corpus(seed: str, items: list[dict], k=10) -> list[str]:
    # use your own data (titles/descriptions that matched the seed) to discover related phrases
    texts = []
    for it in items:
        t = f"{it.get('title','')} {it.get('description','')} {it.get('content','')}"
        if seed.lower() in t.lower():
            texts.append(t)
    if not texts:
        return []
    return top_phrases(texts, topk=k)

# Optional: use pytrends if available
def expand_with_pytrends(seed: str, k=8) -> list[str]:
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([seed], timeframe='now 7-d', geo='US')
        rel = pytrends.related_queries()
        rq = rel.get(seed, {}).get("top")
        if rq is None or rq.empty: return []
        # prefer phrases (exclude exact seed)
        vals = (rq.query.str.lower()
                    .dropna()
                    .pipe(lambda s: s[~s.eq(seed.lower())])
                    .head(k).tolist())
        return vals
    except Exception:
        return []

def _adaptive_pages(seed_result_count: int) -> int:
    # crude heuristic—tune freely
    if seed_result_count >= 200: return min(NEWSAPI_MAX_PAGES, 10)
    if seed_result_count >= 100: return min(NEWSAPI_MAX_PAGES, 6)
    if seed_result_count >= 50:  return min(NEWSAPI_MAX_PAGES, 4)
    return min(NEWSAPI_MAX_PAGES, 2)


# Optional: LLM-driven expansions (DeepSeek)
def expand_with_llm(seed: str, n=12) -> list[str]:
    try:
        prompt = f"""Generate {n} diverse, US-politics-relevant search phrases related to "{seed}" for news discovery. 
Avoid duplicates; include entities, legal cases, agencies, states, and common synonyms. 
Return JSON list only."""
        messages = [{"role": "user", "content": prompt}]
        txt = deepseek_chat(messages, model="deepseek-chat", max_tokens=400, temperature=0.2)
        txt = txt.strip().strip("`").strip()
        obj = json.loads(re.search(r"\[.*\]", txt, re.S).group(0)) if "[" in txt else json.loads(txt)
        # normalize
        outs = []
        for q in obj:
            q = re.sub(r"\s+", " ", str(q)).strip().lower()
            if q and q != seed.lower(): outs.append(q)
        return list(dict.fromkeys(outs))
    except Exception:
        return []


def load_previous_topics() -> List[str]:
    """Load previously identified trending topics to expand our search"""
    try:
        # Change this line:
        topics_file = DATA_DIR / 'trending_topics.json'  # Updated path
        with open(topics_file, 'r') as f:
            data = json.load(f)
            return data.get('topics', [])[:5]
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading previous topics: {str(e)}")
        return []

def save_trending_topics(topics: List[str]):
    """Save trending topics for use in future searches"""
    try:
        # Change this line:
        topics_file = DATA_DIR / 'trending_topics.json'  # Updated path
        with open(topics_file, 'w') as f:
            json.dump({'topics': topics, 'last_updated': dt.datetime.now().isoformat()}, f)
    except Exception as e:
        print(f"Error saving trending topics: {str(e)}")

def process_topic_batch(
    queries: List[Tuple[str, str]],
    window_name: str = "default",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    from_hours: int | None = None,
):
    """
    Process a batch of pre-composed (category, query) pairs with proper rate limiting.
    """
    print(f"Processing {len(queries)} queries in {window_name} window")

    all_content = []
    reddit_client = setup_reddit_client()

    for i, (cat, q) in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: [{cat}] {q}")

        # 1) News — chunk the window to avoid recency bias / per-query caps
        news_items = []
        if start_dt and end_dt:
            # Heuristic: 2–3 day chunks for 2–6 week windows; 1 day if very long and you want max recall
            days = 1
            for ch_start, ch_end in _daterange_chunks(start_dt, end_dt, days=days):
                arts = get_newsapi_articles(
                    q,
                    page_size=min(100, CONFIG["max_items_per_topic"]),
                    max_pages=NEWSAPI_MAX_PAGES,
                    search_in="title,description",
                    from_dt=ch_start,
                    to_dt=ch_end,
                    from_hours=(from_hours if from_hours is not None else 72),
                )
                for a in arts:
                    a["query"] = q
                    a["query_category"] = cat
                    a["categories"] = label_categories_from_taxonomy(
                        a.get("title",""), a.get("description",""), a.get("content","")
                    )
                news_items.extend(arts)
                # be nice to the API
                time.sleep(CONFIG["newsapi_delay"])
        else:
            arts = get_newsapi_articles(
                q,
                page_size=min(100, CONFIG["max_items_per_topic"]),
                max_pages=NEWSAPI_MAX_PAGES,
                search_in="title,description",
                from_dt=None,
                to_dt=None,
                from_hours=(from_hours if from_hours is not None else 72),
            )
            for a in arts:
                a["query"] = q
                a["query_category"] = cat
                a["categories"] = label_categories_from_taxonomy(
                    a.get("title",""), a.get("description",""), a.get("content","")
                )
            news_items.extend(arts)

        # de-dup and append
        news_items = _dedup_items(news_items)
        all_content.extend(news_items)
        print(f"Found {len(news_items)} news articles for query: {q}")

        time.sleep(CONFIG["newsapi_delay"])

        # 2) Reddit (if client available)
        if reddit_client:
            per_sub_total = max(1, CONFIG["max_items_per_topic"] // max(1, len(CONFIG["subreddits_to_monitor"])))
            per_mode = max(1, per_sub_total // 2)

            # Pick a time_filter based on requested window length
            tf = "day"
            wd = _window_days(start_dt, end_dt)
            if wd >= 30:
                tf = "year"
            elif wd >= 7:
                tf = "month"
            elif wd >= 2:
                tf = "week"

            for subreddit in CONFIG["subreddits_to_monitor"]:
                posts = []
                for mode in [("top", tf), ("new", tf)]:
                    posts += get_reddit_posts(reddit_client, subreddit, q, time_filter=mode[1], limit=per_mode)
                # filter + dedup
                seen = set(); dedup = []
                for p in posts:
                    if (start_dt or end_dt) and (not _within_window(p.get("published_at"), start_dt, end_dt)):
                        continue
                    pid = p.get("post_id")
                    if pid and pid not in seen:
                        seen.add(pid)
                        p["query"] = q
                        p["query_category"] = cat
                        p["categories"] = label_categories_from_taxonomy(p.get("title",""), p.get("content",""), "")
                        dedup.append(p)
                all_content.extend(dedup)
                print(f"Found {len(dedup)} Reddit posts in r/{subreddit} for query: {q} (tf={tf})")
                time.sleep(CONFIG["reddit_delay_between_subreddits"])

                # dedup by post_id
                seen = set(); dedup = []
                for p in posts:
                    # drop outside window if a window is set
                    if (start_dt or end_dt) and (not _within_window(p.get("published_at"), start_dt, end_dt)):
                        continue
                    pid = p.get("post_id")
                    if pid and pid not in seen:
                        seen.add(pid)
                        p["query"] = q
                        p["query_category"] = cat
                        p["categories"] = label_categories_from_taxonomy(p.get("title",""), p.get("content",""), "")
                        dedup.append(p)
                all_content.extend(dedup)
                print(f"Found {len(dedup)} Reddit posts in r/{subreddit} for query: {q}")
                time.sleep(CONFIG["reddit_delay_between_subreddits"])

        # 3) Twitter (optional)
        if CONFIG["enable_twitter"]:
            tweets = get_twitter_posts(q, limit=max(1, CONFIG["max_items_per_topic"] // 4))
            for t in tweets:
                t["query"] = q
                t["query_category"] = cat
                t["categories"] = label_categories_from_taxonomy("", t.get("content",""), "")
            all_content.extend(tweets)
            print(f"Found {len(tweets)} tweets for query: {q}")

        time.sleep(CONFIG["reddit_delay_between_topics"])

    return all_content



def daily_collection_job(start_dt: datetime | None = None,
                         end_dt: datetime | None = None,
                         from_hours: int | None = None):
    """Main job to run daily content collection"""
    now = dt.datetime.now()
    print(f"Starting daily content gathering at {now.isoformat()}")

    # 1) Seeds + previous trending
    previous_topics = load_previous_topics()
    base_seeds = list(dict.fromkeys(SEED_TOPICS + previous_topics))

    # 2) Dynamic expansions from recent dumps
    exp = dynamic_expand_topics(
        base_seeds,
        recent_items_json_glob=str(DATA_DIR / "content_*.json"),
        per_seed=6,
        use_pytrends=True,
        use_llm=False
    )

    # 3) Assemble (category, query) pairs:
    #    - taxonomy include terms with NOT-excludes
    #    - seed queries
    #    - expansion queries (tagged as 'seed_expansion')
    pairs: List[Tuple[str, str]] = []
    pairs += taxonomy_queries()
    pairs += seed_queries(base_seeds)
    for s in base_seeds:
        for e in exp.get(s, []):
            pairs.append(("seed_expansion", f'"{e}"'))

    # Deduplicate on query string, keep first category
    seen = set()
    final_pairs: List[Tuple[str, str]] = []
    for cat, q in pairs:
        if q in seen:
            continue
        seen.add(q)
        final_pairs.append((cat, q))

    # Cap daily budget
    final_pairs = final_pairs[:CONFIG["max_topics_per_day"]]

    # Persist an audit trail of queries and expansions (optional but recommended)
    try:
        (DATA_DIR / f"queries_{now:%Y%m%d_%H%M%S}.json").write_text(
            json.dumps({"queries": [{"category": c, "q": q} for c, q in final_pairs]}, indent=2)
        )
        (DATA_DIR / f"expansions_{now:%Y%m%d_%H%M%S}.json").write_text(
            json.dumps(exp, indent=2)
        )
    except Exception as e:
        print(f"[audit] failed to write queries/expansions: {e}")

    topics_to_process = final_pairs
    print(f"Processing {len(topics_to_process)} queries today")

    # 4) Process (ad-hoc if dates/hours provided, else windowed)
    all_content = []

    # If a date window or relative hours is provided: run ad-hoc
    if start_dt or end_dt or (from_hours is not None):
        # If end was passed as a date-only at midnight, treat it as inclusive end-of-day
        if end_dt and end_dt.time() == dt.time(0, 0):
            end_dt = end_dt + dt.timedelta(days=1)

        if start_dt and end_dt:
            # ---- per-day loop ----
            for ch_start, ch_end in _daterange_chunks(start_dt, end_dt, days=1):
                day_str = ch_start.date().isoformat()
                print(f"[ad-hoc] collecting day {day_str} …")

                content_batch = process_topic_batch(
                    topics_to_process,
                    window_name=f"ad_hoc_{day_str}",
                    start_dt=ch_start,
                    end_dt=ch_end,
                    from_hours=None,           # not used when absolute range provided
                )
                content_batch = _dedup_items(content_batch)
                all_content.extend(content_batch)
                all_content = _dedup_items(all_content)

                # save ONE FILE PER DAY
                output_filename = f"content_{day_str}_ad_hoc.json"
                output_path = DATA_DIR / output_filename
                try:
                    with open(output_path, 'w') as f:
                        json.dump({
                            'metadata': {
                                'gathered_at': dt.datetime.now().isoformat(),
                                'content_count': len(content_batch),
                                'topics_used': [q for (_, q) in topics_to_process],
                                'topic_categories_used': [c for (c, _) in topics_to_process],
                                'start_dt': ch_start.isoformat(),
                                'end_dt': ch_end.isoformat(),
                                'from_hours': None
                            },
                            'content': content_batch
                        }, f, indent=2)
                    print(f"Saved {len(content_batch)} items (ad-hoc day) -> {output_path}")
                except Exception as e:
                    print(f"Failed to write ad-hoc day {day_str} output: {e}")
        else:
            # ---- relative window (single batch) ----
            content_batch = process_topic_batch(
                topics_to_process, window_name="ad_hoc",
                start_dt=None, end_dt=None, from_hours=from_hours
            )
            content_batch = _dedup_items(content_batch)
            all_content.extend(content_batch)
            all_content = _dedup_items(all_content)

            output_filename = f"content_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_ad_hoc.json"
            output_path = DATA_DIR / output_filename
            try:
                with open(output_path, 'w') as f:
                    json.dump({
                        'metadata': {
                            'gathered_at': dt.datetime.now().isoformat(),
                            'content_count': len(content_batch),
                            'topics_used': [q for (_, q) in topics_to_process],
                            'topic_categories_used': [c for (c, _) in topics_to_process],
                            'start_dt': None,
                            'end_dt': None,
                            'from_hours': from_hours
                        },
                        'content': content_batch
                    }, f, indent=2)
                print(f"Saved {len(content_batch)} items (ad-hoc relative) -> {output_path}")
            except Exception as e:
                print(f"Failed to write ad-hoc output: {e}")


    else:
        # ORIGINAL windowed behavior
        for i, window in enumerate(CONFIG["collection_windows"]):
            current_hour = dt.datetime.now().hour
            start_h = window["start_hour"]; end_h = start_h + window["duration_hours"]
            if start_h <= current_hour < end_h:
                print(f"Starting collection window {i+1} (hours {start_h}-{end_h})")

                topics_per_window = max(1, CONFIG["max_topics_per_day"] // max(1, len(CONFIG["collection_windows"])))
                start_idx = i * topics_per_window
                end_idx = min(start_idx + topics_per_window, len(topics_to_process))
                window_pairs = topics_to_process[start_idx:end_idx]

                content_batch = process_topic_batch(
                    window_pairs, window_name=f"window_{i+1}",
                    start_dt=None, end_dt=None, from_hours=None
                )
                content_batch = _dedup_items(content_batch)
                all_content.extend(content_batch)
                all_content = _dedup_items(all_content)

                output_filename = f"content_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_window_{i+1}.json"
                output_path = DATA_DIR / output_filename
                try:
                    with open(output_path, 'w') as f:
                        json.dump({
                            'metadata': {
                                'gathered_at': dt.datetime.now().isoformat(),
                                'content_count': len(content_batch),
                                'topics_used': [q for (_, q) in window_pairs],
                                'topic_categories_used': [c for (c, _) in window_pairs]
                            },
                            'content': content_batch
                        }, f, indent=2)
                    print(f"Saved {len(content_batch)} items from window {i+1} -> {output_path}")
                except Exception as e:
                    print(f"Failed to write window {i+1} output: {e}")


    # 5) Trending topics for tomorrow (unchanged)
    word_freq = defaultdict(int)
    for item in all_content:
        text = f"{item.get('title', '')} {item.get('content', '')}"
        words = text.lower().split()
        for word in words:
            if len(word) > 5 and word not in ['http', 'https']:
                word_freq[word] += 1

    trending_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    trending_topics = [word for word, count in trending_words]

    save_trending_topics(trending_topics)
    print(f"Saved {len(trending_topics)} trending topics for next search")

    print(f"Daily collection complete. Gathered {len(all_content)} total items.")


def dynamic_expand_topics(seeds: list[str], recent_items_json_glob: str, per_seed=6, use_pytrends=True, use_llm=False) -> dict[str,list[str]]:
    # Load yesterday/today items to learn co-phrases 
    pat = recent_items_json_glob  # can be absolute or relative
    files = sorted(map(Path, glob(pat)))[-12:]  # last ~few dumps
    recent = []
    for p in files:
        try:
            obj = json.loads(p.read_text())
            recent.extend(obj.get("content", []))
        except Exception:
            pass

    expanded = {}
    for seed in seeds:
        pool = set()
        pool.update(expand_from_corpus(seed, recent, k=per_seed*2))
        if use_pytrends: pool.update(expand_with_pytrends(seed, k=per_seed*2))
        if use_llm:      pool.update(expand_with_llm(seed, n=per_seed*2))
        # keep medium-length phrases; remove the seed itself; cap per_seed
        cleaned = []
        for q in pool:
            q2 = re.sub(r"\s+", " ", q).strip()
            if len(q2) < 4 or q2 == seed.lower(): continue
            cleaned.append(q2)
        # de-dup by lowercase, keep first N
        cleaned = list(dict.fromkeys(cleaned))[:max(1,per_seed)]
        expanded[seed] = cleaned
    return expanded

# ================= SCHEDULING =================
def schedule_daily_collection():
    """Schedule the daily collection job"""
    # Schedule for 2:00 AM daily
    schedule.every().day.at("02:00").do(daily_collection_job)
    
    print("Scheduler started. Daily collection will run at 2:00 AM.")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__" and os.environ.get("RUN_COLLECTOR_ARGS") == "1":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="UTC date start (YYYY-MM-DD or ISO)", default=None)
    ap.add_argument("--end",   help="UTC date end   (YYYY-MM-DD or ISO, exclusive if time given)", default=None)
    ap.add_argument("--from-hours", type=int, default=None,
                    help="Relative window (hours) if you don't pass --start/--end")
    args = ap.parse_args()

    start_dt = _parse_date_utc(args.start)
    end_dt   = _parse_date_utc(args.end)

    # If end was a date-only token (00:00), shift to next midnight to make it inclusive
    if end_dt and end_dt.time() == dt.time(0,0):
        end_dt = end_dt + dt.timedelta(days=1)

    daily_collection_job(start_dt=start_dt, end_dt=end_dt, from_hours=args.from_hours)
    raise SystemExit(0)


if __name__ == "__main__" and os.environ.get("RUN_COLLECTOR_TEST") == "1":
    print("=" * 60)
    print("STARTING TEST RUN WITH MINIMAL CONFIGURATION")
    print("=" * 60)

    TEST_CONFIG = {
        "max_topics_per_day": 40,
        "max_items_per_topic": 15,
        "subreddits_to_monitor": ["politics", "Conservative", "PoliticalDiscussion"],
        "collection_windows": [
            {"start_hour": dt.datetime.now().hour, "duration_hours": 2, "topics": 8},
        ],
        "reddit_delay_between_requests": 5,
        "reddit_delay_between_topics": 3.0,
        "reddit_delay_between_subreddits": 4.0,
        "newsapi_delay": 1.0,
        "max_retries": 2,
        "enable_twitter": False,
    }

    original_config = CONFIG.copy()
    CONFIG.update(TEST_CONFIG)

    try:
        # test run uses default windowed mode
        daily_collection_job()
        print("=" * 60)
        print("TEST RUN COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as e:
        print(f"TEST RUN FAILED WITH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.update(original_config)
