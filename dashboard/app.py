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
import sys
sys.path.append("src")
from analyzer import main as run_analysis

# ── Generate data if database doesn't exist ────────────────────────────────────

if not os.path.exists(DB_PATH):
    with st.spinner("Running analysis for the first time... Please wait."):
        run_analysis()
    st.rerun()

# ── Database path ──────────────────────────────────────────────────────────────

DB_PATH = "data/processed/reddit_analysis.db"

# ── Generate data if database doesn't exist ────────────────────────────────────

if not os.path.exists(DB_PATH):
    st.info("Database not found. Running analysis... This may take a moment.")
    subprocess.run(["python", "src/analyzer.py"], check=True)
    st.rerun()


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Anatomy of Belief",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0a0a0a;
        }
        
        /* Title styling */
        h1 {
            font-family: monospace;
            font-size: 3rem !important;
            font-weight: 900 !important;
            letter-spacing: -2px;
            color: #ffffff !important;
            border-bottom: 3px solid #ff0000;
            padding-bottom: 10px;
        }
        
        /* Headers */
        h2, h3 {
            font-family: monospace;
            color: #ffffff !important;
            border-left: 4px solid #ff0000;
            padding-left: 10px;
        }
        
        /* Metrics */
        [data-testid="stMetric"] {
            background-color: #111111;
            border: 1px solid #333333;
            border-left: 4px solid #ff0000;
            padding: 15px;
            font-family: monospace;
        }
        
        /* Dataframe */
        .stDataFrame {
            border: 1px solid #333333;
        }

        /* Selectbox */
        .stSelectbox {
            font-family: monospace;
        }
    </style>
""", unsafe_allow_html=True)

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
    title='SENTIMENT ANALYSIS',
    labels={'avg_compound': 'AVG SENTIMENT (VADER)', 'category': 'COMMUNITY TYPE'},
    color_discrete_map={'conspiracy': '#ff0000', 'mainstream': '#ffffff'}
)
fig_sentiment.update_layout(
    paper_bgcolor='#0a0a0a',
    plot_bgcolor='#0a0a0a',
    font=dict(family='monospace', color='white'),
    title_font=dict(size=20, color='white'),
    xaxis=dict(gridcolor='#222222', color='white'),
    yaxis=dict(gridcolor='#222222', color='white'),
    showlegend=False
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
    title='CONSPIRACY THEORY KEYWORDS DETECTED',
    labels={'count': 'NUMBER OF POSTS', 'keyword': 'KEYWORD'},
    color_discrete_map={'conspiracy_count': '#ff0000', 'mainstream_count': '#ffffff'}
)
fig_keywords.update_layout(
    paper_bgcolor='#0a0a0a',
    plot_bgcolor='#0a0a0a',
    font=dict(family='monospace', color='white'),
    title_font=dict(size=20, color='white'),
    xaxis=dict(gridcolor='#222222', color='white', tickangle=-45),
    yaxis=dict(gridcolor='#222222', color='white'),
)
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