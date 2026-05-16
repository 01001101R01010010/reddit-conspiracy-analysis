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
    page_icon="",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp { background-color: #fafaf8; }
        
        h1 {
            font-family: monospace !important;
            font-size: 2.8rem !important;
            font-weight: 900 !important;
            letter-spacing: -2px !important;
            color: #1a1a1a !important;
            border-bottom: 4px solid #e8611a;
            padding-bottom: 10px;
            text-transform: uppercase;
        }
        
        h2, h3 {
            font-family: monospace !important;
            color: #1a1a1a !important;
            text-transform: uppercase;
            letter-spacing: -0.5px;
            border-left: 4px solid #e8611a;
            padding-left: 10px;
        }

        p, .stMarkdown p {
            font-family: monospace;
            color: #444444;
        }
        
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1.5px solid #e0e0e0;
            border-top: 4px solid #e8611a;
            padding: 16px;
            font-family: monospace;
        }

        [data-testid="stMetricLabel"] {
            font-family: monospace !important;
            color: #888888 !important;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.7rem !important;
            letter-spacing: 1px;
        }

        [data-testid="stMetricValue"] {
            font-family: monospace !important;
            color: #1a1a1a !important;
            font-size: 1.8rem !important;
        }
        
        .stDataFrame { 
            border: 1.5px solid #e0e0e0;
        }

        .stSelectbox label {
            font-family: monospace !important;
            font-weight: bold;
            text-transform: uppercase;
            color: #1a1a1a;
            font-size: 0.8rem;
            letter-spacing: 1px;
        }

        hr {
            border: none;
            border-top: 1.5px solid #e0e0e0;
            margin: 2rem 0;
        }

        .stCaption {
            font-family: monospace !important;
            color: #999999 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Anatomy of Belief")
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

ORANGE = '#e8611a'
GRAY = '#888888'
LIGHT = '#fafaf8'

CHART_LAYOUT = dict(
    paper_bgcolor=LIGHT,
    plot_bgcolor=LIGHT,
    font=dict(family='monospace', color='#1a1a1a'),
    title_font=dict(size=16, color='#1a1a1a', family='monospace'),
    xaxis=dict(gridcolor='#e0e0e0', color='#444444', linecolor='#cccccc'),
    yaxis=dict(gridcolor='#e0e0e0', color='#444444', linecolor='#cccccc'),
)

# ── Sentiment chart ────────────────────────────────────────────────────────────

st.header("Sentiment Analysis")
st.caption("Proxy = mteb/reddit-clustering (approximate) | Real = pushshift r/conspiracy (authentic)")

if 'dataset' in sentiment_df.columns:
    fig_compare = px.bar(
        sentiment_df,
        x='category',
        y='avg_compound',
        color='dataset',
        barmode='group',
        title='SENTIMENT COMPARISON: PROXY VS REAL DATASET',
        labels={'avg_compound': 'Avg Sentiment (VADER)', 'category': 'Community'},
        color_discrete_map={'proxy': GRAY, 'real': ORANGE}
    )
else:
    fig_compare = px.bar(
        sentiment_df,
        x='category',
        y='avg_compound',
        color='category',
        title='AVERAGE SENTIMENT: CONSPIRACY VS MAINSTREAM',
        labels={'avg_compound': 'Avg Sentiment (VADER)', 'category': 'Community'},
        color_discrete_map={'conspiracy': ORANGE, 'mainstream': GRAY}
    )

fig_compare.update_layout(**CHART_LAYOUT)
st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")

# ── Keyword hits ──────────────────────────────────────────────────────────────

st.header("Conspiracy Theory Keywords")

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
        title='CONSPIRACY THEORY KEYWORDS DETECTED',
        labels={'count': 'Number of Posts', 'keyword': 'Keyword'},
        color_discrete_map={'conspiracy_count': ORANGE, 'mainstream_count': GRAY}
    )
    fig_keywords.update_layout(**CHART_LAYOUT)
    fig_keywords.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_keywords, use_container_width=True)

st.markdown("---")

# ── Raw data ──────────────────────────────────────────────────────────────────

st.header("Raw Data Explorer")

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
st.caption("Data: mteb/reddit-clustering + pushshift-reddit | NLP: VADER + NRCLex | Maciej Reluga")