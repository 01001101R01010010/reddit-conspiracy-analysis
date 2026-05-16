"""
analyzer.py
Analyzes Reddit posts for sentiment, word frequency, bigrams and emotions.
Compares conspiracy-like vs mainstream communities across two datasets.

Limitations:
- Proxy dataset subreddit categorization was done manually
- Bag of Words lacks context (e.g. 'god' could mean 'oh my god')
- Bigrams provide better context than single words
- Future improvement: Topic Modeling or semantic embeddings
"""

import pandas as pd
from collections import Counter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from collections import defaultdict

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
    'moon landing', 'apollo 11', 'nasa fake',
    'nine eleven', 'world trade', 'inside job', 'twin towers', 'jet fuel',
    'flat earth', 'earth flat',
    'illuminati', 'new world', 'world order', 'deep state',
    'climate change', 'global warming',
    'jfk assassination', 'lee harvey', 'kennedy assassination',
    'chemtrails', 'chem trails',
    'qanon',
    'anti vaccine', 'vaccine conspiracy',
    'lizard people', 'reptilian',
    'haarp', 'weather control',
    'zika virus', 'peak oil',
    'anti capitalist', 'new world order',
    'great replacement'
]

EMOTIONS = ['fear', 'anger', 'trust', 'joy', 'sadness', 'disgust', 'anticipation', 'surprise']


def analyze_sentiment(df):
    print("Analyzing sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    df['compound'] = df['title'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    return df


def get_emotions(text):
    try:
        emotion = NRCLex(text)
        emotion.load_raw_text(text)
        return emotion.affect_frequencies
    except:
        return {e: 0.0 for e in EMOTIONS}


def analyze_emotions(df):
    print("Analyzing emotions...")
    for emotion in EMOTIONS:
        df[f'emotion_{emotion}'] = df['title'].apply(
            lambda x: get_emotions(x).get(emotion, 0.0)
        )
    return df


def get_top_words(texts, n=15):
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        all_words.extend(words)
    return Counter(all_words).most_common(n)


def get_top_bigrams(texts, n=15):
    all_bigrams = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        all_bigrams.extend(bigrams)
    return Counter(all_bigrams).most_common(n)


def find_conspiracy_keywords(texts, keywords):
    hits = {kw: 0 for kw in keywords}
    for text in texts:
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                hits[kw] += 1
    return hits


def analyze(df):
    df = analyze_sentiment(df)
    df = analyze_emotions(df)
    return df


def print_summary(df, dataset_name):
    print(f"\n{'='*50}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*50}")

    print("\nSENTIMENT:")
    print(df.groupby('category')['compound'].mean().round(4))

    print("\nEMOTIONS (avg per category):")
    emotion_cols = [f'emotion_{e}' for e in EMOTIONS]
    print(df.groupby('category')[emotion_cols].mean().round(4))

    print("\nTOP 10 BIGRAMS - CONSPIRACY:")
    for (a, b), count in get_top_bigrams(df[df['category'] == 'conspiracy']['title'], 10):
        print(f"  {a} {b}: {count}")

    print("\nTOP 10 BIGRAMS - MAINSTREAM:")
    for (a, b), count in get_top_bigrams(df[df['category'] == 'mainstream']['title'], 10):
        print(f"  {a} {b}: {count}")


if __name__ == "__main__":
    from collector import load_all_data
    from storage import init_database, save_posts, save_sentiment_summary, save_keyword_hits

    df_proxy, df_real = load_all_data()

    print("\nAnalyzing proxy dataset...")
    df_proxy = analyze(df_proxy)
    print_summary(df_proxy, "proxy")

    print("\nAnalyzing real dataset...")
    df_real = analyze(df_real)
    print_summary(df_real, "real")

    print("\nSaving to database...")
    init_database()
    save_posts(pd.concat([df_proxy, df_real]))
    save_sentiment_summary(pd.concat([df_proxy, df_real]))

    print("\nDone!")