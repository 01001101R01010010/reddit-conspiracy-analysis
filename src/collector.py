"""
collector.py
Collects Reddit posts from two sources:
1. mteb/reddit-clustering (HuggingFace proxy dataset)
2. fddemarco/pushshift-reddit (real Reddit posts from r/conspiracy and mainstream subreddits)
"""

import pandas as pd
from datasets import load_dataset


# ── Constants ──────────────────────────────────────────────────────────────────

# Proxy dataset subreddit mapping
PROXY_CONSPIRACY = ['Anarchism.txt', 'collapse.txt', 'Christianity.txt']
PROXY_MAINSTREAM = ['Coffee.txt', 'australia.txt', 'Advice.txt', 'boston.txt', 'chemistry.txt']

# Real dataset subreddit mapping
REAL_CONSPIRACY = ['conspiracy', 'conspiracytheories']
REAL_MAINSTREAM = ['science', 'worldnews', 'news']

# How many posts to stream from pushshift
STREAM_LIMIT = 500000


# ── Proxy Dataset ──────────────────────────────────────────────────────────────

def load_proxy_dataset():
    """Load mteb/reddit-clustering as proxy dataset."""
    print("Loading proxy dataset (mteb/reddit-clustering)...")
    dataset = load_dataset("mteb/reddit-clustering", split="test")

    df = pd.DataFrame({
        'title': dataset[0]['sentences'],
        'subreddit': dataset[0]['labels']
    })

    def categorize(subreddit):
        if subreddit in PROXY_CONSPIRACY:
            return 'conspiracy'
        elif subreddit in PROXY_MAINSTREAM:
            return 'mainstream'
        else:
            return 'other'

    df['category'] = df['subreddit'].apply(categorize)
    df['dataset'] = 'proxy'
    df = df[df['category'] != 'other'].copy()

    print(f"Proxy dataset: {len(df)} posts")
    print(df['category'].value_counts().to_string())
    return df


# ── Real Dataset ───────────────────────────────────────────────────────────────

def load_real_dataset():
    """Load real Reddit posts from pushshift via HuggingFace streaming."""
    print("\nLoading real dataset (pushshift-reddit)...")

    subreddit_map = {}
    for sub in REAL_CONSPIRACY:
        subreddit_map[sub] = 'conspiracy'
    for sub in REAL_MAINSTREAM:
        subreddit_map[sub] = 'mainstream'

    dataset = load_dataset(
        "fddemarco/pushshift-reddit",
        split="train",
        streaming=True
    )

    collected = []
    count = 0

    for post in dataset:
        sub = post.get('subreddit', '')
        if sub in subreddit_map and post.get('title', '').strip():
            collected.append({
                'title': post['title'],
                'subreddit': sub,
                'category': subreddit_map[sub]
            })
        count += 1
        if count % 50000 == 0:
            print(f"  Checked {count} posts, collected {len(collected)}")
        if count >= STREAM_LIMIT:
            break

    df = pd.DataFrame(collected)
    df['dataset'] = 'real'

    print(f"Real dataset: {len(df)} posts")
    print(df['category'].value_counts().to_string())
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def load_all_data():
    """Load both datasets and return them separately."""
    df_proxy = load_proxy_dataset()
    df_real = load_real_dataset()
    return df_proxy, df_real


if __name__ == "__main__":
    df_proxy, df_real = load_all_data()
    print(f"\nTotal proxy posts: {len(df_proxy)}")
    print(f"Total real posts: {len(df_real)}")