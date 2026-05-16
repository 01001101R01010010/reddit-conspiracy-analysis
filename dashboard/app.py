"""
app.py
Streamlit dashboard for Reddit Conspiracy Analysis.
Compares two datasets: proxy (mteb) and real (pushshift Reddit).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os

DB_PATH = "data/processed/reddit_analysis.db"

st.set_page_config(
    page_title="Anatomy of Belief",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp { background-color: #0a0a0a; }
        h1 {
            font-family: monospace;
            font-size: 3rem !important;
            font-weight: 900 !important;
            letter-spacing: -2px;
            color: #ffffff !important;
            border-bottom: 3px solid #ff0000;
            padding-bottom: 10px;
        }
        h2, h3 {
            font-family: monospace;
            color: #ffffff !important;
            border-left: 4px solid #ff0000;
            padding-left: 10px;
        }
        [data-testid="stMetric"] {
            background-color: #111111;
            border: 1px solid #333333;
            border-left: 4px solid #ff0000;
            padding: 15px;
            font-family: monospace;
        }
        .stDataFrame { border: 1px solid #333333; }
        .stSelectbox { font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Anatomy of Belief")
st.subheader("Linguistic and emotional patterns in conspiracy vs mainstream Reddit communities")
st.markdown("---")

@st.cache_data
def load_sentiment():
    return pd.read_csv("data/processed/sentiment_summary.csv")

@st.cache_data
def load_posts():
    return pd.read_csv("data/processed/posts.csv")

@st.cache_data
def load_keywords():
    return pd.read_csv("data/processed/keyword_hits.csv")

sentiment_df = load_sentiment()
posts_df = load_posts()
keywords_df = load_keywords()

# ── Metrics ────────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Posts Analyzed", len(posts_df))
with col2:
    cons = sentiment_df[sentiment_df['category'] == 'conspiracy']['avg_compound'].mean()
    st.metric("Conspiracy Avg Sentiment", round(cons, 4))
with col3:
    main = sentiment_df[sentiment_df['category'] == 'mainstream']['avg_compound'].mean()
    st.metric("Mainstream Avg Sentiment", round(main, 4))
with col4:
    st.metric("Sentiment Gap", round(cons - main, 4))

st.markdown("---")

# ── Dataset comparison ────────────────────────────────────────────────────────

st.header("📊 Sentiment: Proxy vs Real Dataset")
st.caption("Proxy = mteb/reddit-clustering (approximate) | Real = pushshift r/conspiracy (authentic)")

if 'dataset' in sentiment_df.columns:
    fig_compare = px.bar(
        sentiment_df,
        x='category',
        y='avg_compound',
        color='dataset',
        barmode='group',
        title='Sentiment Comparison: Proxy Dataset vs Real Reddit Data',
        labels={'avg_compound': 'Avg Sentiment (VADER)', 'category': 'Community'},
        color_discrete_map={'proxy': '#888888', 'real': '#ff0000'}
    )
else:
    fig_compare = px.bar(
        sentiment_df,
        x='category',
        y='avg_compound',
        color='category',
        title='Average Sentiment: Conspiracy vs Mainstream',
        labels={'avg_compound': 'Avg Sentiment (VADER)', 'category': 'Community'},
        color_discrete_map={'conspiracy': '#ff0000', 'mainstream': '#ffffff'}
    )

fig_compare.update_layout(
    paper_bgcolor='#0a0a0a',
    plot_bgcolor='#0a0a0a',
    font=dict(family='monospace', color='white'),
    title_font=dict(size=18, color='white'),
    xaxis=dict(gridcolor='#222222', color='white'),
    yaxis=dict(gridcolor='#222222', color='white'),
)
st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")

# ── Keyword hits ──────────────────────────────────────────────────────────────

st.header("🔑 Conspiracy Theory Keywords Detected")

keywords_filtered = keywords_df[
    (keywords_df['conspiracy_count'] > 0) | (keywords_df['mainstream_count'] > 0)
].copy()

if not keywords_filtered.empty:
    fig_keywords = px.bar(
        keywords_filtered.melt(
            id_vars='keyword',
            value_vars=['conspiracy_count', 'mainstream_count'],
            var_name='category',
            value_name='count'
        ),
        x='keyword',
        y='count',
        color='category',
        barmode='group',
        title='Conspiracy Theory Keywords: Conspiracy vs Mainstream',
        labels={'count': 'Number of Posts', 'keyword': 'Keyword'},
        color_discrete_map={'conspiracy_count': '#ff0000', 'mainstream_count': '#ffffff'}
    )
    fig_keywords.update_layout(
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(family='monospace', color='white'),
        title_font=dict(size=18, color='white'),
        xaxis=dict(gridcolor='#222222', color='white', tickangle=-45),
        yaxis=dict(gridcolor='#222222', color='white'),
    )
    st.plotly_chart(fig_keywords, use_container_width=True)

st.markdown("---")

# ── Raw data ──────────────────────────────────────────────────────────────────

st.header("📋 Raw Data Explorer")

col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    category_filter = st.selectbox("Filter by category:", ['all', 'conspiracy', 'mainstream'])
with col_filter2:
    if 'dataset' in posts_df.columns:
        dataset_filter = st.selectbox("Filter by dataset:", ['all', 'proxy', 'real'])
    else:
        dataset_filter = 'all'

filtered = posts_df.copy()
if category_filter != 'all':
    filtered = filtered[filtered['category'] == category_filter]
if dataset_filter != 'all' and 'dataset' in filtered.columns:
    filtered = filtered[filtered['dataset'] == dataset_filter]

cols = ['title', 'category', 'compound']
if 'dataset' in filtered.columns:
    cols = ['title', 'category', 'dataset', 'compound']

st.dataframe(filtered[cols].head(100))

st.markdown("---")
st.caption("Data: mteb/reddit-clustering + pushshift-reddit (HuggingFace) | NLP: VADER + NRCLex | Built by Maciej Reluga")