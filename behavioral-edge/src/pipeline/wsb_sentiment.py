"""
src/pipeline/wsb_sentiment.py

Two data paths:

  ApeWisdom (quick path)
  ─────────────────────
  Free REST API, no key required.  Returns top-mentioned tickers across
  WSB and other subreddits, with mention counts and sentiment rank.
  Use this for production daily runs.

  PRAW (rich path)
  ────────────────
  Full Reddit API access.  Pulls raw posts/comments from r/wallstreetbets,
  extracts ticker mentions, and runs FinBERT or VADER sentiment.
  Use this for historical backfill (combined with Kaggle dataset).

Output table: wsb_mentions  (see db.py for full schema)

Historical backfill note
────────────────────────
For dates before the ApeWisdom API history window, load from a
pre-downloaded Kaggle dataset.  See load_from_csv() below.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.db import get_db, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)

_AW_BASE = settings["api"]["apewisdom"]["base_url"]
_SENTIMENT_MODEL = settings["pipeline"]["sentiment_model"]

# Regex to extract uppercase ticker mentions from free text.
# Matches 1-5 capital letters preceded by $ or a word boundary,
# followed by a word boundary.  Excludes common false-positives.
_COMMON_WORDS = {
    "A", "I", "AM", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS",
    "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US",
    "WE", "AT", "DD", "OG", "OP", "PM", "EV", "AI", "CEO", "CFO", "CTO",
    "ATH", "ATM", "ITM", "OTM", "IV", "DTE", "WSB", "SPY", "SPX", "QQQ",
    "ETF", "IPO", "GDP", "CPI", "FED", "SEC", "FDA", "IMO", "FOMO",
}

_TICKER_RE = re.compile(r"(?<!\w)\$([A-Z]{1,5})(?!\w)|(?<!\w)([A-Z]{2,5})(?!\w)")


def extract_tickers(text: str, known_tickers: set[str] | None = None) -> list[str]:
    """
    Extract candidate ticker symbols from *text*.
    If *known_tickers* is supplied, only return symbols present in that set.
    """
    found: set[str] = set()
    for m in _TICKER_RE.finditer(text):
        sym = (m.group(1) or m.group(2)).upper()
        if sym in _COMMON_WORDS:
            continue
        if known_tickers is not None and sym not in known_tickers:
            continue
        found.add(sym)
    return list(found)


# ── Sentiment backends ─────────────────────────────────────────────────────────

def _vader_sentiment(texts: list[str]) -> list[float]:
    """Return compound scores in [-1, 1] using VADER."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


def _finbert_sentiment(texts: list[str]) -> list[float]:
    """Return compound scores in [-1, 1] using FinBERT (requires transformers)."""
    try:
        from transformers import pipeline as hf_pipeline
        model_name = settings["pipeline"]["finbert_model_name"]
        classifier = hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            truncation=True,
            max_length=512,
        )
        scores = []
        for result in classifier(texts):
            label_scores = {r["label"].lower(): r["score"] for r in result}
            compound = label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0)
            scores.append(compound)
        return scores
    except ImportError:
        log.warning("finbert_unavailable_falling_back_to_vader")
        return _vader_sentiment(texts)


def score_texts(texts: list[str]) -> list[float]:
    """Dispatch to the configured sentiment backend."""
    if not texts:
        return []
    if _SENTIMENT_MODEL == "finbert":
        return _finbert_sentiment(texts)
    return _vader_sentiment(texts)


def sentiment_label(score: float) -> str:
    if score > 0.05:
        return "bullish"
    if score < -0.05:
        return "bearish"
    return "neutral"


# ── ApeWisdom path ─────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _aw_get(path: str, params: dict | None = None) -> dict:
    url = f"{_AW_BASE}/{path}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(1.0)  # be a good citizen
    return resp.json()


def fetch_apewisdom(filter_: str = "all-stocks", page: int = 1) -> list[dict]:
    """
    Fetch top mentions from ApeWisdom.
    filter_ options: 'all-stocks', 'wallstreetbets', 'stocks', etc.
    Returns a list of ticker mention dicts.
    """
    try:
        data = _aw_get(f"filter/{filter_}", {"page": page})
        return data.get("results", [])
    except Exception as exc:
        log.warning("apewisdom_failed", error=str(exc))
        return []


def ingest_apewisdom(target_date: str, db_path=None) -> int:
    """
    Pull current ApeWisdom data and store in wsb_mentions.
    Returns number of rows written.

    Note: ApeWisdom only returns *current* data; for historical dates
    this will write today's snapshot tagged with *target_date*.
    For true historical data, use load_from_csv().
    """
    log.info("apewisdom_ingest", date=target_date)

    rows: list[dict] = []
    for page in range(1, 4):   # pull first 3 pages (~75 tickers)
        results = fetch_apewisdom(page=page)
        if not results:
            break
        for r in results:
            ticker = r.get("ticker", "").upper()
            if not ticker:
                continue
            mentions = r.get("mentions", 0)
            mentions_24h = r.get("mentions_24h_ago", 0)
            velocity = (
                ((mentions - mentions_24h) / mentions_24h * 100)
                if mentions_24h and mentions_24h > 0
                else 0.0
            )
            rank = r.get("rank")
            rank_prior = r.get("rank_24h_ago")
            rank_change = (rank_prior - rank) if (rank and rank_prior) else None

            rows.append({
                "date": target_date,
                "ticker": ticker,
                "mention_count": mentions,
                "mention_count_24h_prior": mentions_24h,
                "mention_velocity": velocity,
                "sentiment_score": None,   # ApeWisdom doesn't provide raw score
                "sentiment_label": "bullish" if r.get("sentiment", "") == "Bullish" else
                                   "bearish" if r.get("sentiment", "") == "Bearish" else "neutral",
                "upvotes": r.get("upvotes"),
                "rank": rank,
                "rank_change_24h": rank_change,
                "source": "apewisdom",
            })

    with get_db(db_path) as conn:
        upsert_rows(conn, "wsb_mentions", rows)

    log.info("apewisdom_done", date=target_date, rows=len(rows))
    return len(rows)


# ── PRAW path ──────────────────────────────────────────────────────────────────

def _build_praw_client():
    """Construct and return an authenticated praw.Reddit instance."""
    import praw
    cfg = settings["api"]["reddit"]
    return praw.Reddit(
        client_id=cfg["client_id"],
        client_secret=cfg["client_secret"],
        user_agent=cfg["user_agent"],
        username=cfg.get("username"),
        password=cfg.get("password"),
    )


def fetch_wsb_posts(
    subreddit: str = "wallstreetbets",
    limit: int = 500,
    before: float | None = None,
) -> list[dict]:
    """
    Fetch recent posts from *subreddit* using PRAW.
    *before* is a Unix timestamp to paginate backwards in time.
    Returns list of post dicts with: id, created_utc, title, selftext, score.
    """
    try:
        reddit = _build_praw_client()
        sub = reddit.subreddit(subreddit)
        posts = []
        for submission in sub.new(limit=limit):
            if before and submission.created_utc >= before:
                continue
            posts.append({
                "id": submission.id,
                "created_utc": submission.created_utc,
                "title": submission.title,
                "selftext": submission.selftext,
                "score": submission.score,
            })
        return posts
    except Exception as exc:
        log.error("praw_fetch_failed", subreddit=subreddit, error=str(exc))
        return []


def ingest_praw(
    target_date: str,
    known_tickers: set[str] | None = None,
    db_path=None,
) -> int:
    """
    Pull posts from configured subreddits, extract ticker mentions,
    run sentiment, and store in wsb_mentions.

    For historical data, prefer load_from_csv() which is much faster.
    """
    log.info("praw_ingest", date=target_date)
    subreddits: list[str] = settings["pipeline"]["wsb_subreddits"]
    target_dt = datetime.fromisoformat(target_date)
    start_ts = target_dt.timestamp()
    end_ts   = (target_dt + timedelta(days=1)).timestamp()

    # Accumulate mentions per ticker: {ticker: [texts]}
    ticker_texts: dict[str, list[str]] = {}
    ticker_scores: dict[str, list[int]] = {}

    for sub in subreddits:
        posts = fetch_wsb_posts(subreddit=sub, limit=500)
        for post in posts:
            if not (start_ts <= post["created_utc"] < end_ts):
                continue
            text = f"{post['title']} {post['selftext']}"
            tickers = extract_tickers(text, known_tickers)
            for t in tickers:
                ticker_texts.setdefault(t, []).append(text)
                ticker_scores.setdefault(t, []).append(post["score"])

    if not ticker_texts:
        log.info("praw_no_data", date=target_date)
        return 0

    rows: list[dict] = []
    for ticker, texts in ticker_texts.items():
        scores_arr = score_texts(texts)
        avg_score = sum(scores_arr) / len(scores_arr) if scores_arr else 0.0
        total_upvotes = sum(ticker_scores.get(ticker, []))

        rows.append({
            "date": target_date,
            "ticker": ticker,
            "mention_count": len(texts),
            "mention_count_24h_prior": None,  # requires prior-day query
            "mention_velocity": None,
            "sentiment_score": avg_score,
            "sentiment_label": sentiment_label(avg_score),
            "upvotes": total_upvotes,
            "rank": None,
            "rank_change_24h": None,
            "source": "praw",
        })

    # Fill in prior-day mention counts and velocity
    _backfill_velocity(rows, target_date, db_path)

    with get_db(db_path) as conn:
        upsert_rows(conn, "wsb_mentions", rows)

    log.info("praw_done", date=target_date, tickers=len(rows))
    return len(rows)


def _backfill_velocity(rows: list[dict], target_date: str, db_path) -> None:
    """Mutate *rows* in place to fill mention_count_24h_prior and mention_velocity."""
    from src.utils.db import fetch_df, get_connection
    prior = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    tickers = [r["ticker"] for r in rows]
    if not tickers:
        return
    placeholders = ",".join("?" * len(tickers))
    conn = get_connection(db_path)
    prior_df = fetch_df(
        conn,
        f"SELECT ticker, mention_count FROM wsb_mentions WHERE date = ? AND ticker IN ({placeholders})",
        tuple([prior] + tickers),
    )
    prior_map = dict(zip(prior_df["ticker"], prior_df["mention_count"])) if not prior_df.empty else {}
    for row in rows:
        prior_count = prior_map.get(row["ticker"])
        row["mention_count_24h_prior"] = prior_count
        if prior_count and prior_count > 0 and row["mention_count"] is not None:
            row["mention_velocity"] = (row["mention_count"] - prior_count) / prior_count * 100
        else:
            row["mention_velocity"] = 0.0


# ── Historical CSV loader (Kaggle / Reddit data dump) ─────────────────────────

def load_from_csv(
    csv_path: str,
    known_tickers: set[str] | None = None,
    db_path=None,
    text_col: str = "body",
    date_col: str = "created_utc",
    score_col: str = "score",
) -> int:
    """
    Parse a Reddit data CSV/Parquet (e.g. from Kaggle WSB dataset) and
    populate wsb_mentions.

    Expected columns: *date_col* (Unix timestamp or ISO date), *text_col*,
    optionally *score_col*.

    Returns total rows written.
    """
    log.info("csv_load_start", path=csv_path)

    if csv_path.endswith(".parquet"):
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path, low_memory=False)

    # Normalize date column
    if pd.api.types.is_numeric_dtype(df[date_col]):
        df["_date"] = pd.to_datetime(df[date_col], unit="s").dt.date
    else:
        df["_date"] = pd.to_datetime(df[date_col]).dt.date

    total_written = 0

    for day, group in df.groupby("_date"):
        day_str = day.isoformat()
        ticker_texts: dict[str, list[str]] = {}
        ticker_scores: dict[str, list[int]] = {}

        for _, row_ in group.iterrows():
            text = str(row_.get(text_col, ""))
            score = int(row_.get(score_col, 0)) if score_col in group.columns else 0
            for t in extract_tickers(text, known_tickers):
                ticker_texts.setdefault(t, []).append(text)
                ticker_scores.setdefault(t, []).append(score)

        if not ticker_texts:
            continue

        rows: list[dict] = []
        for ticker, texts in ticker_texts.items():
            scores_arr = score_texts(texts)
            avg_score = sum(scores_arr) / len(scores_arr) if scores_arr else 0.0
            rows.append({
                "date": day_str,
                "ticker": ticker,
                "mention_count": len(texts),
                "mention_count_24h_prior": None,
                "mention_velocity": None,
                "sentiment_score": avg_score,
                "sentiment_label": sentiment_label(avg_score),
                "upvotes": sum(ticker_scores[ticker]),
                "rank": None,
                "rank_change_24h": None,
                "source": "kaggle_csv",
            })

        with get_db(db_path) as conn:
            upsert_rows(conn, "wsb_mentions", rows)
        total_written += len(rows)

    # Back-fill velocity after all dates are loaded
    _compute_velocity_bulk(db_path)

    log.info("csv_load_done", path=csv_path, rows=total_written)
    return total_written


def _compute_velocity_bulk(db_path=None) -> None:
    """
    Compute mention_velocity for all wsb_mentions rows that have NULL velocity.
    Run once after a bulk CSV load.
    """
    from src.utils.db import fetch_df, get_connection
    conn = get_connection(db_path)
    df = fetch_df(conn, "SELECT date, ticker, mention_count FROM wsb_mentions ORDER BY ticker, date")
    if df.empty:
        return

    df["prev_count"] = df.groupby("ticker")["mention_count"].shift(1)
    df["mention_velocity"] = (
        (df["mention_count"] - df["prev_count"]) / df["prev_count"].replace(0, float("nan")) * 100
    ).fillna(0.0)
    df["mention_count_24h_prior"] = df["prev_count"]

    rows = df[["date", "ticker", "mention_count_24h_prior", "mention_velocity"]].to_dict("records")
    with get_db(db_path) as conn:
        for row in rows:
            conn.execute(
                """
                UPDATE wsb_mentions
                   SET mention_count_24h_prior = ?,
                       mention_velocity        = ?
                 WHERE date = ? AND ticker = ?
                """,
                (row["mention_count_24h_prior"], row["mention_velocity"], row["date"], row["ticker"]),
            )
    log.info("velocity_bulk_computed", rows=len(rows))
