"""
storage.py
Saves analysis results to DuckDB database.
DuckDB is a lightweight, fast SQL database - perfect for analytics pipelines.
"""

import duckdb
import pandas as pd
from datetime import datetime


# ── Constants ──────────────────────────────────────────────────────────────────

DB_PATH = "data/processed/reddit_analysis.db"


# ── Database Setup ─────────────────────────────────────────────────────────────

def init_database():
    """Create database tables if they don't exist."""
    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            title VARCHAR,
            subreddit VARCHAR,
            category VARCHAR,
            compound FLOAT,
            run_date DATE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_summary (
            category VARCHAR,
            avg_compound FLOAT,
            post_count INTEGER,
            run_date DATE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS keyword_hits (
            keyword VARCHAR,
            conspiracy_count INTEGER,
            mainstream_count INTEGER,
            run_date DATE
        )
    """)

    con.close()
    print(f"Database initialized: {DB_PATH}")


# ── Save Functions ─────────────────────────────────────────────────────────────

def save_posts(df):
    """Save analyzed posts to database."""
    con = duckdb.connect(DB_PATH)
    today = datetime.today().date()

    df_to_save = df[['title', 'subreddit', 'category', 'compound']].copy()
    df_to_save['run_date'] = today

    con.execute("INSERT INTO posts SELECT * FROM df_to_save")
    con.close()
    print(f"Saved {len(df_to_save)} posts to database")


def save_sentiment_summary(df):
    """Save average sentiment per category to database."""
    con = duckdb.connect(DB_PATH)
    today = datetime.today().date()

    summary = df.groupby('category').agg(
        avg_compound=('compound', 'mean'),
        post_count=('compound', 'count')
    ).reset_index()
    summary['run_date'] = today

    con.execute("INSERT INTO sentiment_summary SELECT * FROM summary")
    con.close()
    print("Saved sentiment summary to database")


def save_keyword_hits(results_df):
    """Save conspiracy keyword hits to database."""
    con = duckdb.connect(DB_PATH)
    today = datetime.today().date()

    results_df['run_date'] = today
    con.execute("INSERT INTO keyword_hits SELECT * FROM results_df")
    con.close()
    print("Saved keyword hits to database")


# ── Read Functions ─────────────────────────────────────────────────────────────

def get_sentiment_summary():
    """Read sentiment summary from database."""
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM sentiment_summary ORDER BY run_date DESC").df()
    con.close()
    return df


def get_keyword_hits():
    """Read keyword hits from database."""
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM keyword_hits ORDER BY conspiracy_count DESC").df()
    con.close()
    return df


if __name__ == "__main__":
    init_database()
    print("Storage module ready!")