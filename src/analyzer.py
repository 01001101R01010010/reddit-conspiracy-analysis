"""
analyzer.py
Analyzes Reddit posts for sentiment and linguistic patterns.
Compares conspiracy-like vs mainstream communities.

Limitations:
- Subreddit categorization was done manually
- Bag of Words lacks context (e.g. 'god' could mean 'oh my god')
- Bigrams provide better context than single words
- Future improvement: Topic Modeling or semantic embeddings
"""

import pandas as pd
from collections import Counter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datasets import load_dataset
from storage import init_database, save_posts, save_sentiment_summary, save_keyword_hits


# ── Constants ──────────────────────────────────────────────────────────────────

CONSPIRACY_SUBREDDITS = ['Anarchism.txt', 'collapse.txt', 'Christianity.txt']
MAINSTREAM_SUBREDDITS = ['Coffee.txt', 'australia.txt', 'Advice.txt', 'boston.txt', 'chemistry.txt']

STOP_WORDS = set([
    'the', 'and', 'for', 'you', 'are', 'with', 'what', 'this', 'that',
    'how', 'from', 'about', 'not', 'have', 'can', 'all', 'was', 'been',
    'will', 'his', 'her', 'they', 'their', 'our', 'your', 'its', 'but',
    'just', 'more', 'when', 'who', 'which', 'has', 'had', 'did', 'one',
    'out', 'get', 'got', 'why', 'any', 'now', 'new', 'than', 'into',
    'after', 'over', 'should', 'would', 'could', 'being', 'my', 'me',
    'him', 'she', 'we', 'it', 'is', 'in', 'of', 'to', 'a', 'be', 'do',
    'boston', 'coffee', 'australia', 'australian', 'chemistry', 'com',
    'there', 'some', 'like', 'post', 'also', 'very', 'use', 'using',
    'used', 'make', 'made', 'even', 'much', 'many', 'most', 'other',
    'such', 'then', 'them', 'they', 'these'
])

CONSPIRACY_KEYWORDS = [
    # Moon landing / Apollo 11
    'moon landing', 'apollo 11', 'nasa fake',
    # 9/11 / World Trade Center
    'nine eleven', 'world trade', 'inside job', 'twin towers', 'jet fuel',
    # Flat Earth
    'flat earth', 'earth flat',
    # Illuminati / New World Order
    'illuminati', 'new world', 'world order', 'deep state',
    # Climate change conspiracy
    'climate change', 'global warming',
    # JFK
    'jfk assassination', 'lee harvey', 'kennedy assassination',
    # Chemtrails
    'chemtrails', 'chem trails',
    # QAnon
    'qanon',
    # Vaccines
    'anti vaccine', 'vaccine conspiracy',
    # Reptilians
    'lizard people', 'reptilian',
    # HAARP
    'haarp', 'weather control',
    # Zika
    'zika virus',
    # Peak Oil
    'peak oil',
    # Anti-system
    'anti capitalist', 'new world order',
    # Great Replacement
    'great replacement'
]


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data():
    """Load Reddit clustering dataset from HuggingFace."""
    print("Loading dataset...")
    dataset = load_dataset("mteb/reddit-clustering", split="test")
    df = pd.DataFrame({
        'title': dataset[0]['sentences'],
        'subreddit': dataset[0]['labels']
    })
    print(f"Loaded {len(df)} posts from {df['subreddit'].nunique()} subreddits")
    return df


# ── Categorization ─────────────────────────────────────────────────────────────

def categorize(subreddit):
    """
    Assign category based on subreddit name.
    Note: categorization is manual - known limitation of this dataset.
    """
    if subreddit in CONSPIRACY_SUBREDDITS:
        return 'conspiracy'
    elif subreddit in MAINSTREAM_SUBREDDITS:
        return 'mainstream'
    else:
        return 'other'


# ── Sentiment Analysis ─────────────────────────────────────────────────────────

def analyze_sentiment(df):
    """Add VADER compound sentiment score to each post."""
    print("Analyzing sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    df['compound'] = df['title'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    return df


# ── Word Frequency ─────────────────────────────────────────────────────────────

def get_top_words(texts, n=20):
    """Return top N words after removing stop words."""
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        all_words.extend(words)
    return Counter(all_words).most_common(n)


def get_top_bigrams(texts, n=15):
    """Return top N bigrams after removing stop words."""
    all_bigrams = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        all_bigrams.extend(bigrams)
    return Counter(all_bigrams).most_common(n)


# ── Conspiracy Keywords ────────────────────────────────────────────────────────

def find_conspiracy_keywords(texts, keywords):
    """Count occurrences of known conspiracy theory keywords in posts."""
    hits = {kw: 0 for kw in keywords}
    for text in texts:
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                hits[kw] += 1
    return hits


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load and prepare data
    df = load_data()
    df['category'] = df['subreddit'].apply(categorize)
    df = analyze_sentiment(df)

    # Filter out 'other' category
    df = df[df['category'] != 'other'].copy()
    print(f"Filtered dataset: {len(df)} posts\n")

    # ── Sentiment comparison
    print("=" * 50)
    print("SENTIMENT ANALYSIS")
    print("=" * 50)
    print(df.groupby('category')['compound'].mean().round(4))

    # ── Word frequency
    print("\n" + "=" * 50)
    print("TOP 15 WORDS: CONSPIRACY")
    print("=" * 50)
    for word, count in get_top_words(df[df['category'] == 'conspiracy']['title'], 15):
        print(f"  {word}: {count}")

    print("\n" + "=" * 50)
    print("TOP 15 WORDS: MAINSTREAM")
    print("=" * 50)
    for word, count in get_top_words(df[df['category'] == 'mainstream']['title'], 15):
        print(f"  {word}: {count}")

    # ── Bigrams
    print("\n" + "=" * 50)
    print("TOP 15 BIGRAMS: CONSPIRACY")
    print("=" * 50)
    for (a, b), count in get_top_bigrams(df[df['category'] == 'conspiracy']['title'], 15):
        print(f"  {a} {b}: {count}")

    print("\n" + "=" * 50)
    print("TOP 15 BIGRAMS: MAINSTREAM")
    print("=" * 50)
    for (a, b), count in get_top_bigrams(df[df['category'] == 'mainstream']['title'], 15):
        print(f"  {a} {b}: {count}")

    # ── Conspiracy keywords
    print("\n" + "=" * 50)
    print("CONSPIRACY THEORY KEYWORDS DETECTED")
    print("=" * 50)
    conspiracy_hits = find_conspiracy_keywords(
        df[df['category'] == 'conspiracy']['title'], CONSPIRACY_KEYWORDS
    )
    mainstream_hits = find_conspiracy_keywords(
        df[df['category'] == 'mainstream']['title'], CONSPIRACY_KEYWORDS
    )

    results = pd.DataFrame({
        'keyword': CONSPIRACY_KEYWORDS,
        'conspiracy': [conspiracy_hits[kw] for kw in CONSPIRACY_KEYWORDS],
        'mainstream': [mainstream_hits[kw] for kw in CONSPIRACY_KEYWORDS]
    }).sort_values('conspiracy', ascending=False)

    print(results.to_string(index=False))

    # ── Save to database
    print("\n" + "=" * 50)
    print("SAVING TO DATABASE")
    print("=" * 50)
    init_database()
    save_posts(df)
    save_sentiment_summary(df)
    save_keyword_hits(results)

    print("\nAnalysis complete!")
    return df


if __name__ == "__main__":
    df = main()