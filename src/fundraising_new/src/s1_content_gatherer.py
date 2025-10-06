import os
import json
import datetime
import time
import requests
import praw
import snscrape.modules.twitter as sntwitter
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


# Calculate the path to the .env file
current_dir = Path(__file__).parent
env_path = current_dir.parent.parent / "secret.env"

NEWSAPI_MAX_PAGES = int(os.getenv("NEWSAPI_MAX_PAGES", "1"))  # default 1 for free tier

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
reddit_limiter = RateLimiter(55)  # Stay under 60 RPM
newsapi_limiter = RateLimiter(30)  # NewsAPI free tier is 100 requests/day

# ================= CORE FUNCTIONS =================
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

def get_newsapi_articles(query: str, page_size: int = 100, max_pages: int = 1, search_in="title,description") -> List[Dict[str, Any]]:

    """Fetch articles from NewsAPI.org for a given query."""
    newsapi_limiter.wait_if_needed()
    
    articles = []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'sortBy': 'publishedAt',
            'apiKey': NEWSAPI_KEY,
            'pageSize': page_size,
            'language': 'en'
        }

        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"NewsAPI error for '{query}': {response.status_code}")
            return articles
            
        data = response.json()

        if data['status'] == 'ok':
            print(f"Found {data['totalResults']} total results for '{query}'")
            for article in data['articles']:
                if article.get('content') in [None, '[Removed]', '[+ chars]']:
                    continue
                    
                content = article.get('content', '')
                if '[+' in content and 'chars]' in content:
                    content = article.get('description', content)
                
                articles.append({
                    'source': 'newsapi',
                    'title': article.get('title', ''),
                    'content': content,
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source_name': article.get('source', {}).get('name', ''),
                    'author': article.get('author', ''),
                    'query': query
                })
        else:
            print(f"NewsAPI error for query '{query}': {data.get('message', 'Unknown error')}")

    except Exception as e:
        print(f"General exception fetching news for {query}: {str(e)}")

    return articles

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
    since_date = (datetime.datetime.now() - datetime.timedelta(days=since_days)).strftime('%Y-%m-%d')
    
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
            json.dump({'topics': topics, 'last_updated': datetime.datetime.now().isoformat()}, f)
    except Exception as e:
        print(f"Error saving trending topics: {str(e)}")

def process_topic_batch(queries: List[str], window_name: str = "default"):
    """Process a batch of pre-composed query strings with proper rate limiting"""
    print(f"Processing {len(queries)} queries in {window_name} window")

    all_content = []
    reddit_client = setup_reddit_client()

    for i, q in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {q}")

        # 1) News: favor diversity (pageSize=100, max_pages=NEWSAPI_MAX_PAGES)
        articles = get_newsapi_articles(
            q,
            page_size=min(100, CONFIG["max_items_per_topic"]),
            max_pages=NEWSAPI_MAX_PAGES,
            search_in="title,description"
        )
        all_content.extend(articles)
        print(f"Found {len(articles)} news articles for query: {q}")
        time.sleep(CONFIG["newsapi_delay"])

        # 2) Reddit: split budget between 'top' and 'new' in a 1-week window, then dedup
        if reddit_client:
            per_sub_total = max(1, CONFIG["max_items_per_topic"] // max(1, len(CONFIG["subreddits_to_monitor"])))
            per_mode = max(1, per_sub_total // 2)
            for subreddit in CONFIG["subreddits_to_monitor"]:
                posts = []
                for mode in [("top","week"), ("new","week")]:
                    posts += get_reddit_posts(reddit_client, subreddit, q, time_filter=mode[1], limit=per_mode)
                # dedup by post_id
                seen = set(); dedup = []
                for p in posts:
                    pid = p.get("post_id")
                    if pid and pid not in seen:
                        seen.add(pid); dedup.append(p)
                all_content.extend(dedup)
                print(f"Found {len(dedup)} Reddit posts in r/{subreddit} for query: {q}")
                time.sleep(CONFIG["reddit_delay_between_subreddits"])

        # 3) Twitter (optional)
        if CONFIG["enable_twitter"]:
            tweets = get_twitter_posts(q, limit=max(1, CONFIG["max_items_per_topic"] // 4))
            all_content.extend(tweets)
            print(f"Found {len(tweets)} tweets for query: {q}")

        time.sleep(CONFIG["reddit_delay_between_topics"])

    return all_content


def daily_collection_job():
    """Main job to run daily content collection"""
    print(f"Starting daily content gathering at {datetime.datetime.now().isoformat()}")
    
    # Load previous trending topics to expand our search
    previous_topics = load_previous_topics()
    base_seeds = SEED_TOPICS + previous_topics

    # Dynamic expansions → a set of distinct query strings
    exp = dynamic_expand_topics(
        base_seeds,
        recent_items_json_glob=str(DATA_DIR / "content_*.json"),
        per_seed=6,
        use_pytrends=True,
        use_llm=False
    )

    # Flatten and interleave: each seed’s own query + its expansions as separate queries
    all_queries = []
    for s in base_seeds:
        all_queries.append(f'"{s}"')  # the seed itself as a query
        for e in exp.get(s, []):
            all_queries.append(f'"{e}"')

    # Dedup and cap to your daily budget
    all_queries = list(dict.fromkeys(all_queries))[:CONFIG["max_topics_per_day"]]

    # Fallback if no expansions (still run seeds)
    if not all_queries:
        all_queries = [f'"{s}"' for s in base_seeds][:CONFIG["max_topics_per_day"]]

    topics_to_process = all_queries
    print(f"Processing {len(topics_to_process)} queries today")

    
    all_content = []
    
    # Process topics in batches based on collection windows
    for i, window in enumerate(CONFIG["collection_windows"]):
        current_hour = datetime.datetime.now().hour
        
        # Check if we should process this window now
        if current_hour >= window["start_hour"] and current_hour < window["start_hour"] + window["duration_hours"]:
            print(f"Starting collection window {i+1} (hours {window['start_hour']}-{window['start_hour'] + window['duration_hours']})")
            
            # Calculate which topics to process in this window
            topics_per_window = CONFIG["max_topics_per_day"] // len(CONFIG["collection_windows"])
            start_idx = i * topics_per_window
            end_idx = min(start_idx + topics_per_window, len(topics_to_process))
            window_topics = topics_to_process[start_idx:end_idx]
            
            # Process this batch of topics
            content_batch = process_topic_batch(window_topics, f"window_{i+1}")
            all_content.extend(content_batch)
            
            # Save intermediate results
            output_filename = f"content_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_window_{i+1}.json"
            output_path = DATA_DIR / output_filename
            with open(output_path, 'w') as f:
                json.dump({
                    'metadata': {
                        'gathered_at': datetime.datetime.now().isoformat(),
                        'content_count': len(content_batch),
                        'topics_used': window_topics
                    },
                    'content': content_batch
                }, f, indent=2)
            
            print(f"Saved {len(content_batch)} items from window {i+1}")
    
    # 5. Extract trending topics from today's content for tomorrow's search
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

def build_news_queries(seed: str, expansions: list[str]) -> list[str]:
    # Build NewsAPI queries that bias to titles but stay generic (no hand-crafted OR lists)
    qs = []
    # the seed itself
    qs.append(f'"{seed}"')
    # each expansion as its own query (more diversity, better than paging on a single query)
    for e in expansions:
        qs.append(f'"{e}"')
    return qs

# ================= SCHEDULING =================
def schedule_daily_collection():
    """Schedule the daily collection job"""
    # Schedule for 2:00 AM daily
    schedule.every().day.at("02:00").do(daily_collection_job)
    
    print("Scheduler started. Daily collection will run at 2:00 AM.")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING TEST RUN WITH MINIMAL CONFIGURATION")
    print("=" * 60)
    
    # Temporary test configuration - minimal settings for quick verification
    TEST_CONFIG = {
        "max_topics_per_day": 40,           # Increased from 3 to 8
        "max_items_per_topic": 15,         # Increased from 5 to 10
        "subreddits_to_monitor": ["politics", "Conservative", "PoliticalDiscussion"],  # Added more subreddits
        "collection_windows": [
            {"start_hour": datetime.datetime.now().hour, "duration_hours": 2, "topics": 8},
        ],
        "reddit_delay_between_requests": 5,  # Slightly longer delays
        "reddit_delay_between_topics": 3.0,
        "reddit_delay_between_subreddits": 4.0,
        "newsapi_delay": 1.0,
        "max_retries": 2,
        "enable_twitter": False,
    }
    
    # Temporarily replace the main config with test config
    original_config = CONFIG.copy()
    CONFIG.update(TEST_CONFIG)
    
    # Run the collection job immediately
    try:
        daily_collection_job()
        print("=" * 60)
        print("TEST RUN COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as e:
        print(f"TEST RUN FAILED WITH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Restore original configuration
    CONFIG.update(original_config)