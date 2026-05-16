# 🔍 Anatomy of Belief
### Linguistic and Emotional Patterns in Conspiracy vs Mainstream Reddit Communities

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reddit-conspiracy-analysis-wxdpzrewtedqnxejbywsx7.streamlit.app/)

---

## Overview

This project is an NLP data pipeline that analyzes linguistic and emotional patterns in Reddit communities. It compares conspiracy-like communities with mainstream ones to answer the question:

> **Do conspiracy communities speak a different emotional language than mainstream Reddit?**

**Key finding:** Conspiracy communities show consistently more negative sentiment (avg: -0.058) compared to mainstream communities (avg: +0.036), and use distinctly different vocabulary centered around systemic and apocalyptic themes.

---

## Live Dashboard

👉 [View Live Dashboard](https://reddit-conspiracy-analysis-wxdpzrewtedqnxejbywsx7.streamlit.app/)

> Note: App may take ~30s to load on first visit (Streamlit Cloud free tier)

---

## Tech Stack

| Technology | Role |
|---|---|
| Python + Pandas | Data processing |
| HuggingFace Datasets | Data source |
| VADER | Sentiment analysis (NLP) |
| DuckDB | Data storage (SQL) |
| Plotly | Interactive visualizations |
| Streamlit | Public dashboard |
| GitHub Actions | Pipeline automation (daily) |
| Docker | Containerization |

---

## Project Structure

```
reddit-conspiracy-analysis/
├── src/
│   ├── analyzer.py        # NLP pipeline
│   └── storage.py         # DuckDB storage
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb  # Exploratory data analysis
├── data/
│   └── processed/         # DuckDB database + CSV exports
├── .github/
│   └── workflows/
│       └── pipeline.yml   # GitHub Actions automation
└── Dockerfile             # Container configuration
```

---

## Key Findings

- **Sentiment:** Conspiracy communities (-0.058) are more negative than mainstream (+0.036)
- **Top conspiracy bigrams:** `climate change`, `global warming`, `peak oil`, `donald trump`, `zika virus`
- **Top mainstream bigrams:** `need help`, `french press`, `periodic table`, `espresso machine`
- **Conspiracy keywords detected:** Climate change and environmental topics dominate

---

## Data Limitations & Known Issues

This project was built under real-world constraints. Here's an honest assessment:

**Data quality:**
- Dataset covers 2012 Reddit data — the conspiracy landscape has changed significantly since (QAnon, COVID narratives, Pizzagate did not yet exist)
- Class imbalance: 890 conspiracy posts vs 9,796 mainstream posts (11:1 ratio) — results should be interpreted with caution
- Reddit API access was attempted but not obtained due to current API restrictions — pushshift archive was used as an alternative

**Analytical limitations:**
- Sentiment differences are small (-0.11 vs -0.03) and lack formal statistical significance testing — a future version should include p-values
- NRCLex emotion detection runs on short post titles (avg 8-10 words) which limits reliability
- Subreddit categorization (conspiracy vs mainstream) was done manually — a subjective decision

**What I would do with more time:**
- Obtain live Reddit API access for current data
- Balance the dataset through undersampling or weighting
- Add statistical significance tests (Mann-Whitney U test)
- Use sentence embeddings instead of Bag of Words for deeper semantic analysis
- Expand to full post text, not just titles

> Note: Some conspiracy theories have historically proven true. Negative sentiment may reflect genuine institutional distrust rather than irrationality.

---

## How to Run

### Local
```bash
git clone https://github.com/01001101R01010010/reddit-conspiracy-analysis.git
cd reddit-conspiracy-analysis
pip install -r requirements.txt
streamlit run dashboard/app.py
```

### Docker
```bash
docker build -t reddit-conspiracy-analysis .
docker run -p 8501:8501 reddit-conspiracy-analysis
```

---

## Automation

GitHub Actions runs the pipeline automatically every day at 3:00 AM UTC, updating the database with fresh analysis results.

---

*Built as part of a Data Engineering portfolio project. Background: Finance → Data Engineering transition.*