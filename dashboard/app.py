"""
app.py
Streamlit dashboard for Reddit Conspiracy Analysis.
Reads data from DuckDB and displays interactive visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
import sys
import os

# ── Database path ──────────────────────────────────────────────────────────────

DB_PATH = "data/processed/reddit_analysis.db"

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Anatomy of Belief",
    page_icon="🔍",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🔍 Anatomy of Belief")
st.subheader("Linguistic and emotional patterns in conspiracy vs mainstream Reddit communities")
st.markdown("---")

# ── Load data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_sentiment_summary():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM sentiment_summary ORDER BY run_date DESC").df()
    con.close()
    return df

@st.cache_data
def load_posts():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM posts").df()
    con.close()
    return df

@st.cache_data
def load_keyword_hits():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM keyword_hits ORDER BY conspiracy_count DESC").df()
    con.close()
    return df

sentiment_df = load_sentiment_summary()
posts_df = load_posts()
keywords_df = load_keyword_hits()

# ── Metrics ────────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Posts Analyzed", len(posts_df))

with col2:
    conspiracy_sentiment = sentiment_df[sentiment_df['category'] == 'conspiracy']['avg_compound'].values[0]
    st.metric("Conspiracy Avg Sentiment", round(conspiracy_sentiment, 4))

with col3:
    mainstream_sentiment = sentiment_df[sentiment_df['category'] == 'mainstream']['avg_compound'].values[0]
    st.metric("Mainstream Avg Sentiment", round(mainstream_sentiment, 4))

st.markdown("---")

# ── Sentiment chart ────────────────────────────────────────────────────────────

st.header("📊 Sentiment Analysis")

fig_sentiment = px.bar(
    sentiment_df,
    x='category',
    y='avg_compound',
    color='category',
    title='Average Sentiment Score: Conspiracy vs Mainstream',
    labels={'avg_compound': 'Average Sentiment (VADER)', 'category': 'Community Type'},
    color_discrete_map={'conspiracy_count': 'red', 'mainstream_count': 'blue'}
)
st.plotly_chart(fig_sentiment, use_container_width=True)

st.markdown("---")

# ── Keyword hits ───────────────────────────────────────────────────────────────

st.header("🔑 Conspiracy Theory Keywords")

keywords_filtered = keywords_df[
    (keywords_df['conspiracy_count'] > 0) | (keywords_df['mainstream_count'] > 0)
].copy()

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
    color_discrete_map={'conspiracy_count': 'red', 'mainstream_count': 'blue'}
)
fig_keywords.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_keywords, use_container_width=True)

st.markdown("---")

# ── Raw data ───────────────────────────────────────────────────────────────────

st.header("📋 Raw Data")

category_filter = st.selectbox(
    "Filter by category:",
    options=['all', 'conspiracy_count', 'mainstream_count']
)

if category_filter == 'all':
    st.dataframe(posts_df[['title', 'category', 'compound']].head(100))
else:
    filtered = posts_df[posts_df['category'] == category_filter]
    st.dataframe(filtered[['title', 'category', 'compound']].head(100))

st.markdown("---")
st.caption("Data source: mteb/reddit-clustering (HuggingFace) | NLP: VADER Sentiment Analysis")