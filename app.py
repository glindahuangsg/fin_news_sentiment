"""
Financial News Sentiment Monitor for Bank of China  (ISOM5240 group project)

The app chains TWO Hugging Face pipelines:

    news text
        -> pipeline 1: sentiment  (DistilBERT fine-tuned in 01_finetune_sentiment_model.ipynb)
        -> pipeline 2: topic      (pre-trained BERT, selected in 03_test_and_experiments.ipynb)
        -> alert rule: negative + confident  ->  priority and team routing from the topic

Run locally:   streamlit run app.py
"""

import hashlib
import html
import os
import random
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline

# ============================================================
# Configuration
# ============================================================
COMPANY_NAME = "Bank of China"
COMPANY_URL = "https://www.boc.cn/en/"

# Pipeline 1 model: DistilBERT fine-tuned in notebook 01 and published to the Hugging Face Hub in notebook 02.
SENTIMENT_MODEL = "glindahuangsg/boc-finsentiment-distilbert"
# Pipeline 2 model: the pre-trained topic classifier selected in notebook 03 (Experiment 2).
TOPIC_MODEL = "leonas5555/finnews-topic-single-classify"

MAX_LENGTH = 128         # same as in the notebooks
BATCH_SIZE = 16
MAX_ITEMS = 80           # most news items scored at once (keeps the free CPU responsive)
NEWS_QUERY = '"Bank of China"'   # default search for the live news feed
# Produced by notebook 03. Optional: if it sits next to app.py, the Model check page offers it.
TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_test_sample.csv")

SENTIMENTS = ["Negative", "Neutral", "Positive"]
# Diverging colours: red and blue stay distinguishable for colour-blind readers
# (red/green does not). Neutral is grey.
SENTIMENT_COLORS = {"Negative": "#e34948", "Neutral": "#898781", "Positive": "#2a78d6"}
PRIORITY_ICONS = {"High": "🔴", "Medium": "🟠"}

# The 20 topics pipeline 2 can return, and the team each topic is routed to.
TOPIC_TEAMS = {
    "Legal | Regulation": "Compliance & Legal",
    "Earnings": "Finance & Investor Relations",
    "Financials": "Finance & Investor Relations",
    "Dividend": "Finance & Investor Relations",
    "Analyst Update": "Finance & Investor Relations",
    "Stock Commentary": "Finance & Investor Relations",
    "Stock Movement": "Finance & Investor Relations",
    "IPO": "Strategy & Corporate Development",
    "M&A | Investments": "Strategy & Corporate Development",
    "Macro": "Treasury & Markets",
    "Markets": "Treasury & Markets",
    "Fed | Central Banks": "Treasury & Markets",
    "Currencies": "Treasury & Markets",
    "Treasuries | Corporate Debt": "Treasury & Markets",
    "Energy | Oil": "Treasury & Markets",
    "Gold | Metals | Materials": "Treasury & Markets",
    "Personnel Change": "Corporate Communications",
    "Company | Product News": "Corporate Communications",
    "General News | Opinion": "Corporate Communications",
    "Politics": "Corporate Communications",
}
TOPIC_NAMES = list(TOPIC_TEAMS)
DEFAULT_TEAM = "Corporate Communications"
DEFAULT_PRIORITY_TOPICS = ["Legal | Regulation", "Earnings", "Financials", "Treasuries | Corporate Debt"]

# Who a news item is about. Checked top to bottom; the first match wins.
DEMO_ENTITY = "Our Bank (demo)"
WATCHLIST = [
    (COMPANY_NAME, ["bank of china", "boc hong kong", "bochk"]),
    (DEMO_ENTITY, ["our bank"]),
    ("Peer bank", ["icbc", "agbank", "psbc", "eximbank", "china construction bank", "bank of communications",
                   "hsbc", "standard chartered", "competitor"]),
    ("Regulator", ["pboc", "nfra", "csrc", "hkma", "regulator", "central bank"]),
]
# Several other institutions have names that END in "Bank of China". They are
# replaced by a short code before matching, otherwise every one of them would
# be tagged as the company.
OTHER_BANKS_OF_CHINA = {
    "people's bank of china": "pboc", "people\u2019s bank of china": "pboc",
    "industrial and commercial bank of china": "icbc", "agricultural bank of china": "agbank",
    "postal savings bank of china": "psbc", "export-import bank of china": "eximbank",
}


# ============================================================
# Text cleaning (identical to the notebooks)
# ============================================================
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
CASHTAG_RE = re.compile(r"\$(?=[A-Za-z])")
SPACE_RE = re.compile(r"\s+")


# Clean a text (links, @mentions, extra spaces) the same way as in training.
def clean_text(text):
    """
    Tidy a piece of text before it goes to the models.

    The steps match the cleaning used when the sentiment model was trained, so the
    model sees the same kind of input it learned from:
      1. turn HTML codes such as &amp; back into normal characters,
      2. remove web links and @mentions,
      3. drop the "$" in front of stock tickers ($AAPL -> AAPL),
      4. collapse repeated spaces and trim the ends.

    Args:
        text: any value; it is converted to a string first.
    Returns:
        The cleaned string (can be empty, e.g. when the text was only a link).
    """
    text = html.unescape(str(text))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = CASHTAG_RE.sub("", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


# ============================================================
# The two Hugging Face pipelines
# ============================================================
# st.cache_resource keeps each model in memory, so it is loaded once and not on
# every click.
# Pipeline 1: load the fine-tuned sentiment model once and keep it in memory.
@st.cache_resource(show_spinner="Loading pipeline 1: sentiment model...")
def load_sentiment_pipeline():
    """
    Pipeline 1: load our fine-tuned DistilBERT sentiment model from the Hugging Face Hub.

    The model is downloaded the first time and then kept in memory by
    st.cache_resource, so later clicks reuse it instead of loading it again.

    Returns:
        A Hugging Face text-classification pipeline that outputs negative / neutral / positive.
    """
    return pipeline("text-classification", model=SENTIMENT_MODEL, device=-1)   # -1 = CPU


# Pipeline 2: load the pre-trained topic model once and keep it in memory.
@st.cache_resource(show_spinner="Loading pipeline 2: topic model...")
def load_topic_pipeline():
    """
    Pipeline 2: load the pre-trained financial-news topic classifier from the Hugging Face Hub.

    Like pipeline 1, it is loaded once and cached in memory.

    Returns:
        A Hugging Face text-classification pipeline that outputs one of the 20 topics in TOPIC_TEAMS.
    """
    return pipeline("text-classification", model=TOPIC_MODEL, device=-1)


SENTIMENT_ALIASES = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive",
                     "bearish": "Negative", "bullish": "Positive"}


# Convert any model's label name into Negative / Neutral / Positive.
def to_sentiment(label):
    """
    Translate a model's label name into one of the app's three sentiments.

    Different models name their classes differently ("negative", "Bearish", ...).
    SENTIMENT_ALIASES maps all of them to Negative / Neutral / Positive, and any
    unknown label falls back to Neutral so the app never crashes on a new name.

    Args:
        label: the label string returned by the model.
    Returns:
        "Negative", "Neutral" or "Positive".
    """
    return SENTIMENT_ALIASES.get(str(label).strip().lower(), "Neutral")


# Convert the topic model's label (e.g. LABEL_3) into a readable topic name.
def to_topic(label):
    """
    Turn the topic model's label into a readable topic name.

    Some models return generic labels such as "LABEL_3" instead of a name. In that
    case the number is used as a position in TOPIC_NAMES ("LABEL_3" -> 4th topic).
    A label that already is a name is returned unchanged.

    Args:
        label: the label string returned by the topic model.
    Returns:
        The topic name, e.g. "Earnings".
    """
    label = str(label).strip()
    match = re.fullmatch(r"LABEL_(\d+)", label)
    if match and int(match.group(1)) < len(TOPIC_NAMES):
        return TOPIC_NAMES[int(match.group(1))]
    return label


# Run both pipelines on a batch of texts; one result row per text (core of the app).
@st.cache_data(show_spinner="Analysing news with both pipelines...")
def score_texts(texts):
    """
    Run both pipelines over a batch of texts. This is the core of the app.

    Each text is cleaned, then:
      - pipeline 1 gives the probability of each sentiment (negative/neutral/positive),
      - pipeline 2 gives the most likely topic and its score.
    The result is cached by st.cache_data, so the same headlines are only scored once.

    Args:
        texts: a TUPLE of strings (a tuple, not a list, because Streamlit's cache
               needs an unchangeable input to recognise texts it has seen before).
    Returns:
        A DataFrame with one row per text and the columns: sentiment, confidence,
        p_negative, p_neutral, p_positive, topic, topic_confidence.
    """
    cleaned = [clean_text(t) or "(empty)" for t in texts]
    sentiment_pipe = load_sentiment_pipeline()
    topic_pipe = load_topic_pipeline()

    # Pipeline 1. top_k=None returns the score of every class, not only the best one.
    sent_out = sentiment_pipe(cleaned, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH, top_k=None)
    # Pipeline 2. The default returns the best topic only.
    topic_out = topic_pipe(cleaned, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH)

    rows = []
    for class_scores, topic in zip(sent_out, topic_out):
        probabilities = {s: 0.0 for s in SENTIMENTS}
        for item in class_scores:
            probabilities[to_sentiment(item["label"])] = float(item["score"])
        best = max(probabilities, key=probabilities.get)
        rows.append({
            "sentiment": best,
            "confidence": round(probabilities[best], 4),
            "p_negative": round(probabilities["Negative"], 4),
            "p_neutral": round(probabilities["Neutral"], 4),
            "p_positive": round(probabilities["Positive"], 4),
            "topic": to_topic(topic["label"]),
            "topic_confidence": round(float(topic["score"]), 4),
        })
    return pd.DataFrame(rows)


# ============================================================
# News sources
# ============================================================
# Current time in UTC, so all timestamps in the app are comparable.
def utc_now():
    """
    Return the current time in UTC, without a time-zone label.

    All timestamps in the app (live news, uploads, demo) are converted to this
    same form, so they can be compared and sorted with each other.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


# News source 1: simulated headlines for demonstrating the app.
@st.cache_data
def demo_news(n=53):
    """
    News source 1: build a set of simulated headlines for demonstrations.

    Headlines are made from templates such as "{e} posts 5% rise in quarterly net
    profit", filled in with "Our Bank", "Competitor A" or "Competitor B", plus a few
    sector-wide headlines. A fixed random seed (42) makes the set the same every run.
    No real company is named, and the app labels these headlines as simulated.

    Args:
        n: how many headlines to return.
    Returns:
        A DataFrame with the columns title, source, timestamp (within the last 3 days) and url (empty).
    """
    rng = random.Random(42)
    sources = ["Newswire A", "Newswire B", "Business Daily", "Market Journal", "Finance Weekly"]
    banks = ["Our Bank", "Competitor A", "Competitor B"]
    bank_templates = [                       # {e} is replaced by one of the banks
        "Regulator fines {e} over anti-money-laundering control failures",
        "{e} faces probe into mortgage lending practices",
        "{e} reports a rise in non-performing loans in the third quarter",
        "Data breach reported at a branch of {e}",
        "{e} net interest margin narrows as rates fall",
        "Customer complaints surge over wealth products sold by {e}",
        "{e} shares slide after profit misses analyst forecasts",
        "{e} posts 5% rise in quarterly net profit",
        "{e} beats earnings estimates on strong loan growth",
        "{e} raises its dividend after record annual results",
        "Analysts upgrade {e} on improved asset quality",
        "{e} wins mandate as lead underwriter for green bond issue",
        "{e} launches a new digital banking platform",
        "{e} to publish interim results next week",
        "{e} appoints a new chief risk officer",
        "{e} holds its annual general meeting",
    ]
    sector_headlines = [
        "Central bank keeps the benchmark lending rate unchanged",
        "New capital adequacy rules announced for commercial banks",
        "Central bank cuts the reserve requirement ratio by 25 basis points",
        "Banking sector lending growth steady in September",
        "Regulator tightens scrutiny of wealth management products",
    ]
    titles = [t.format(e=bank) for t in bank_templates for bank in banks] + sector_headlines
    rng.shuffle(titles)
    now = utc_now()
    rows = [{"title": title, "source": rng.choice(sources),
             "timestamp": now - timedelta(minutes=rng.randint(5, 72 * 60)), "url": ""}
            for title in titles[:n]]
    return pd.DataFrame(rows)


# Convert a downloaded RSS feed into a table of headlines.
def parse_rss(xml_bytes):
    """
    Turn a downloaded RSS feed (XML) into a table of headlines.

    For every <item> in the feed it reads the title, publisher, publication date
    and link. Google News adds " - Publisher" to the end of each title, so that
    part is removed. If a date cannot be read, the current time is used instead.

    Args:
        xml_bytes: the raw bytes of the RSS feed.
    Returns:
        A DataFrame with the columns title, source, timestamp and url.
    """
    root = ET.fromstring(xml_bytes)
    rows = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(" - " + source):       # "Headline - Publisher" -> "Headline"
            title = title[: -len(" - " + source)].strip()
        try:
            stamp = parsedate_to_datetime(item.findtext("pubDate") or "")
            stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
        except (TypeError, ValueError):
            stamp = utc_now()
        if title:
            rows.append({"title": title, "source": source or "Unknown", "timestamp": stamp,
                         "url": (item.findtext("link") or "").strip()})
    return pd.DataFrame(rows, columns=["title", "source", "timestamp", "url"])


# News source 2: fetch current headlines from Google News (cached 15 minutes).
@st.cache_data(ttl=900, show_spinner="Fetching live headlines...")
def live_news(query):
    """
    News source 2: fetch the latest real headlines from the public Google News RSS feed.

    The search phrase is put into the feed's web address, the feed is downloaded
    (with a 15-second time limit) and parsed by parse_rss. The result is cached for
    15 minutes (ttl=900 seconds) so the feed is not downloaded on every click;
    the "Refresh now" button in the sidebar clears this cache.

    Args:
        query: the search phrase, e.g. '"Bank of China"'.
    Returns:
        A DataFrame with the columns title, source, timestamp and url.
    Raises:
        An error if the network or the feed is unavailable (handled in choose_news).
    """
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (student news-monitor project)"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return parse_rss(response.read())


# News source 3: read headlines from a CSV file uploaded by the user.
def uploaded_news(file):
    """
    News source 3: read headlines from a CSV file uploaded by the user.

    Only a 'title' (or 'text') column is required. The optional columns 'source',
    'timestamp' (or 'published' / 'date') and 'url' are used when present; missing
    values get sensible defaults. Column names are matched without regard to case.

    Args:
        file: the uploaded file object from st.file_uploader.
    Returns:
        A DataFrame with the columns title, source, timestamp and url (empty titles removed).
    Raises:
        ValueError if the file has no 'title' or 'text' column.
    """
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "title" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "title"})
    if "title" not in df.columns:
        raise ValueError(f"The CSV needs a column named 'title' or 'text'. Found: {list(df.columns)}")
    out = pd.DataFrame({"title": df["title"].astype(str).str.strip()})
    out["source"] = df["source"].fillna("Unknown").astype(str) if "source" in df.columns else "Uploaded file"
    stamp_col = next((c for c in ("timestamp", "published", "date") if c in df.columns), None)
    stamps = pd.to_datetime(df[stamp_col], errors="coerce") if stamp_col else pd.Series(pd.NaT, index=df.index)
    out["timestamp"] = stamps.fillna(pd.Timestamp(utc_now()))
    out["url"] = df["url"].fillna("").astype(str) if "url" in df.columns else ""
    return out[out["title"].str.len() > 0].reset_index(drop=True)


# Decide by keywords who a headline is about (company, peer bank, regulator, other).
def tag_entity(title):
    """
    Decide by keywords who a headline is about.

    This is a simple keyword rule, not a model. The WATCHLIST is checked from top
    to bottom and the first group with a matching keyword wins.

    Args:
        title: the headline text.
    Returns:
        "Bank of China", "Our Bank (demo)", "Peer bank", "Regulator" or "Other".
    """
    low = str(title).lower()                       # compare in lower case
    # Replace other banks whose names end in "Bank of China" (e.g. People's Bank of China)
    # by a short code first, so they are not mistaken for the company.
    for full_name, code in OTHER_BANKS_OF_CHINA.items():
        low = low.replace(full_name, code)
    # The first watchlist group with a matching keyword decides the entity.
    for name, keywords in WATCHLIST:
        if any(k in low for k in keywords):
            return name
    return "Other"                                 # no keyword matched


# Add model results, entity, team and an id to the news table.
def enrich(news):
    """
    Add everything the pages need to the raw news table.

    Steps: keep at most MAX_ITEMS headlines, make all timestamps UTC, run both
    pipelines (score_texts), tag who each headline is about (tag_entity), look up
    the team for each topic, and give each headline a short stable id.

    Args:
        news: a DataFrame with the columns title, source, timestamp and url.
    Returns:
        The same table plus the model columns, entity, team and id, newest first.
    """
    # Limit the number of items so scoring stays fast on a free CPU.
    news = news.head(MAX_ITEMS).reset_index(drop=True)
    # Make every timestamp a UTC time; unreadable dates become "now".
    stamps = pd.to_datetime(news["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    news["timestamp"] = stamps.fillna(pd.Timestamp(utc_now()))
    # Run both pipelines and put their results next to the headlines.
    scores = score_texts(tuple(news["title"]))
    out = pd.concat([news, scores], axis=1)
    # Who the headline is about, and which team should handle its topic.
    out["entity"] = out["title"].map(tag_entity)
    out["team"] = out["topic"].map(TOPIC_TEAMS).fillna(DEFAULT_TEAM)
    # A short id built from the title, so an alert keeps its status when the page reloads.
    # lambda t: the first 10 characters of the title's MD5 fingerprint, used as the id
    out["id"] = out["title"].map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()[:10])
    return out.sort_values("timestamp", ascending=False).reset_index(drop=True)


# ============================================================
# Alerts: where the two pipelines meet
# ============================================================
# The starting alert rules (confidence threshold, priority topics and entities).
def default_rules():
    """
    Return the starting alert rules, used until the user changes them in the Alert centre.

    Returns:
        A dictionary with:
          - min_confidence:    how sure the sentiment model must be that an item is negative (0.85),
          - priority_topics:   topics that make an alert High priority,
          - priority_entities: entities that make an alert High priority.
    """
    return {"min_confidence": 0.85, "priority_topics": list(DEFAULT_PRIORITY_TOPICS),
            "priority_entities": [COMPANY_NAME, DEMO_ENTITY]}


# Turn confident negative items into alerts with a priority and a team.
def build_alerts(news, rules):
    """
    Combine the two pipelines into alerts. This is where pipeline 1 and pipeline 2 meet.

    Alert    = pipeline 1 says Negative with confidence at or above the threshold.
    Priority = High when pipeline 2's topic is on the priority list, or the item is
               about a watched entity; otherwise Medium.
    Team     = taken from pipeline 2's topic (already added in enrich).

    Args:
        news:  the enriched (and filtered) news table.
        rules: the alert rules (see default_rules).
    Returns:
        The alert items with extra columns priority and status, High priority first, newest first.
    """
    # Step 1: keep only confident negative items.
    alerts = news[(news["sentiment"] == "Negative") & (news["confidence"] >= rules["min_confidence"])].copy()
    # Step 2: High priority if the topic OR the entity is on the priority lists.
    is_high = alerts["topic"].isin(rules["priority_topics"]) | alerts["entity"].isin(rules["priority_entities"])
    alerts["priority"] = is_high.map({True: "High", False: "Medium"})
    # Step 3: status set by the user in the Alert centre (Open if never changed).
    # lambda i: look up the status saved for this alert id, "Open" if none was saved
    alerts["status"] = alerts["id"].map(lambda i: st.session_state["alert_status"].get(i, "Open"))
    # Step 4: sort High before Medium, and newest first within each.
    order = alerts["priority"].map({"High": 0, "Medium": 1})
    return alerts.assign(_order=order).sort_values(["_order", "timestamp"], ascending=[True, False]).drop(columns="_order")


# ============================================================
# Small display helpers
# ============================================================
# A coloured dot plus the sentiment name, for display.
def dot(sentiment):
    """
    Return HTML for a coloured dot followed by the sentiment name, e.g. "● Negative".

    The word is always shown next to the colour, so the meaning is clear even for
    readers who cannot tell the colours apart.
    """
    return f"<span style='color:{SENTIMENT_COLORS.get(sentiment, '#898781')};'>●</span> {sentiment}"


# Escape Markdown characters so headlines are shown exactly as written.
def md_safe(text):
    """
    Escape special characters so outside text is displayed exactly as written.

    Headlines come from outside the app. Streamlit renders text as Markdown, so
    characters such as * _ $ # would change the formatting (for example,
    "$5 billion ... $3" would be shown as a maths formula). A backslash is put
    in front of each such character.

    Args:
        text: the text to display.
    Returns:
        The escaped text, safe to pass to st.markdown.
    """
    return re.sub(r"([\\`*_\[\]$#<>|~])", r"\\\1", str(text))


# Format a timestamp as readable text.
def when(stamp):
    """Format a timestamp as readable text, e.g. "2026-10-12 08:30"."""
    return pd.Timestamp(stamp).strftime("%Y-%m-%d %H:%M")


# Link to the original article, or a note when there is none.
def source_link(url):
    """
    Return an HTML link to the original article, opening in a new tab.

    Demo headlines have no real article, so for them (or any empty / non-web
    address) the short note "no original link" is returned instead.
    """
    url = str(url or "")
    if url.startswith(("http://", "https://")):
        return f"<a href='{html.escape(url, quote=True)}' target='_blank'>Open original article</a>"
    return "no original link"


# Apply the same look to every chart.
def base_layout(fig, height):
    """
    Apply the same layout to every Plotly chart, for a consistent look.

    Sets the height, small margins, rounded bars and a horizontal legend above the chart.

    Args:
        fig:    the Plotly figure to style.
        height: chart height in pixels.
    Returns:
        The same figure, styled.
    """
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), barcornerradius=4,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="", traceorder="normal"),
        hoverlabel=dict(namelength=-1),
    )
    return fig


# Draw one news item as a card with its sentiment, topic and details.
def news_card(row, show_details=True):
    """
    Draw one news item as a bordered card.

    The card shows the headline, then its sentiment, topic, source, time and a link.
    An optional "Details" section shows who it is about, the suggested team and the
    models' scores for each class.

    Args:
        row:          one row of the enriched news table.
        show_details: whether to include the "Details" section.
    """
    with st.container(border=True):
        st.markdown(f"**{md_safe(row['title'])}**")
        st.markdown(
            f"{dot(row['sentiment'])} &nbsp;·&nbsp; {html.escape(str(row['topic']))} &nbsp;·&nbsp; "
            f"{html.escape(str(row['source']))} &nbsp;·&nbsp; {when(row['timestamp'])} UTC &nbsp;·&nbsp; "
            f"{source_link(row.get('url', ''))}",
            unsafe_allow_html=True,
        )
        if show_details:
            with st.expander("Details"):
                st.write(f"About: **{row['entity']}** · Suggested team: **{row['team']}**")
                st.caption(
                    f"Model view — negative {row['p_negative']:.0%}, neutral {row['p_neutral']:.0%}, "
                    f"positive {row['p_positive']:.0%}; topic match {row['topic_confidence']:.0%}. "
                    "These are the models' own scores, not a guarantee of correctness."
                )


# ============================================================
# Sidebar: data source and filters
# ============================================================
# Sidebar: pick the news source and load the headlines.
def choose_news():
    """
    Sidebar: let the user pick where the news comes from, and load it.

    Three sources are offered:
      - Live headlines from Google News (falls back to demo headlines if the feed fails),
      - Demo headlines (simulated),
      - an uploaded CSV file (falls back to demo headlines until a file is given).

    Returns:
        (news, note): the raw news table, and a short sentence describing the
        source that is shown at the top of the pages.
    """
    st.sidebar.header("News source")
    source = st.sidebar.radio(
        "Where the news comes from",
        ["Live headlines (Google News)", "Demo headlines (simulated)", "Upload a CSV"],
        key="source",
    )
    note = ""
    if source.startswith("Live"):
        query = st.sidebar.text_input("Search phrase", NEWS_QUERY, key="query")
        if st.sidebar.button("Refresh now"):
            live_news.clear()
        try:
            news = live_news(query)
            if news.empty:
                raise ValueError("the feed returned no items")
            # All headlines are kept, as in the demo and CSV modes. The search also returns other banks whose
            # names contain the phrase (e.g. People's Bank of China); the "About" filter in the sidebar
            # narrows the view to the company, peer banks or regulators.
            note = (f"Live headlines for {query}, via Google News RSS. "
                    f"Use the 'About' filter to narrow to {COMPANY_NAME}, peer banks or regulators.")
        except Exception as error:      # network problem, feed change, empty result
            st.sidebar.warning(f"Live headlines are unavailable ({type(error).__name__}). Showing demo headlines instead.")
            news, note = demo_news(), "Demo headlines (simulated), because the live feed could not be reached."
    elif source.startswith("Upload"):
        file = st.sidebar.file_uploader("CSV with a 'title' or 'text' column", type=["csv"], key="news_file")
        if file is None:
            st.sidebar.info("No file yet. Showing demo headlines.")
            news, note = demo_news(), "Demo headlines (simulated). Upload a CSV in the sidebar to analyse your own."
        else:
            try:
                news = uploaded_news(file)
                note = f"Uploaded file: {file.name} ({len(news)} items)."
            except Exception as error:
                st.sidebar.error(f"Could not read the file: {error}")
                news, note = demo_news(), "Demo headlines (simulated), because the upload could not be read."
    else:
        news, note = demo_news(), "Demo headlines (simulated for demonstration; not real news)."
    if len(news) > MAX_ITEMS:
        note += f" Showing the first {MAX_ITEMS} of {len(news)} items."
    return news, note


# Sidebar: show the filters and return the matching news.
def filter_news(news):
    """
    Sidebar: show the filters (period, sentiment, topic, about) and apply them.

    The Topic and About lists are built from the news that is loaded, so they only
    offer values that actually occur. An empty Topic or About selection means "all".

    Args:
        news: the enriched news table.
    Returns:
        The rows that pass every filter.
    """
    st.sidebar.header("Filters")
    period = st.sidebar.radio("Period", ["Last 24 hours", "Last 7 days", "All"], index=2, key="period")
    sentiments = st.sidebar.multiselect("Sentiment", SENTIMENTS, default=SENTIMENTS, key="f_sentiment")
    topics_all = sorted(news["topic"].unique())
    topics = st.sidebar.multiselect("Topic", topics_all, default=[], key="f_topic", placeholder="All topics")
    entities_all = sorted(news["entity"].unique())
    entities = st.sidebar.multiselect("About", entities_all, default=[], key="f_entity", placeholder="Everyone")

    out = news
    # Period: keep items newer than 24 hours or 7 days ago.
    if period != "All":
        hours = 24 if period == "Last 24 hours" else 24 * 7
        out = out[out["timestamp"] >= utc_now() - timedelta(hours=hours)]
    # Sentiment: keep the ticked sentiments.
    out = out[out["sentiment"].isin(sentiments)]
    # Topic and About: only filter when something is selected (empty = everything).
    if topics:
        out = out[out["topic"].isin(topics)]
    if entities:
        out = out[out["entity"].isin(entities)]
    return out.reset_index(drop=True)


# ============================================================
# Page: Dashboard
# ============================================================
# Page: overview with key numbers, charts, urgent alerts and latest news.
def page_dashboard(news, alerts, note):
    """
    Page "Dashboard": the overview.

    Shows, from top to bottom: four key numbers, the sentiment mix bar, sentiment
    over time, negative items by topic, the top open high-priority alerts and a
    table of the latest news.

    Args:
        news:   the filtered news table.
        alerts: the alerts built from that news.
        note:   the sentence describing the news source.
    """
    st.title(f"🏦 {COMPANY_NAME} · News Sentiment Monitor")
    st.caption(f"{note} Updated {when(utc_now())} UTC.")

    total = len(news)
    negative = int((news["sentiment"] == "Negative").sum())
    positive = int((news["sentiment"] == "Positive").sum())
    open_high = int(((alerts["priority"] == "High") & (alerts["status"] == "Open")).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("News items", total)
    c2.metric("Negative", negative, help="Items the sentiment model classified as negative.")
    c3.metric("Share negative", f"{(negative / total if total else 0):.0%}")
    c4.metric("Open high-priority alerts", open_high,
              help="Negative items the model is confident about, on a priority topic or about a watched entity.")

    if total == 0:
        st.info("No news items match the current filters.")
        return

    st.subheader("Sentiment mix")
    counts = news["sentiment"].value_counts().reindex(SENTIMENTS).fillna(0).astype(int)
    fig = go.Figure()
    for s in SENTIMENTS:
        fig.add_bar(y=["All items"], x=[counts[s]], name=s, orientation="h", marker_color=SENTIMENT_COLORS[s],
                    text=[f"{s} {counts[s]} ({counts[s] / total:.0%})"], textposition="inside",
                    insidetextanchor="middle", hovertemplate=f"{s}: %{{x}} items<extra></extra>")
    fig.update_layout(barmode="stack", uniformtext=dict(minsize=11, mode="hide"), showlegend=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(base_layout(fig, 120), key="mix")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment over time")
        span_hours = (news["timestamp"].max() - news["timestamp"].min()).total_seconds() / 3600
        bucket = "h" if span_hours <= 30 else "D"
        trend = (news.assign(bucket=news["timestamp"].dt.floor(bucket))
                 .groupby(["bucket", "sentiment"]).size().unstack(fill_value=0).reindex(columns=SENTIMENTS, fill_value=0))
        fig = go.Figure()
        for s in SENTIMENTS:
            fig.add_bar(x=trend.index, y=trend[s], name=s, marker_color=SENTIMENT_COLORS[s],
                        hovertemplate=f"{s}: %{{y}}<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.25, hovermode="x unified")
        fig.update_yaxes(title_text="Items per " + ("hour" if bucket == "h" else "day"), rangemode="tozero")
        st.plotly_chart(base_layout(fig, 320), key="trend")
    with right:
        st.subheader("Negative items by topic")
        by_topic = news[news["sentiment"] == "Negative"]["topic"].value_counts().head(8).sort_values()
        if by_topic.empty:
            st.info("No negative items in the current selection.")
        else:
            fig = go.Figure(go.Bar(x=by_topic.values, y=by_topic.index, orientation="h",
                                   marker_color=SENTIMENT_COLORS["Negative"], text=by_topic.values, width=0.55,
                                   textposition="outside", hovertemplate="%{y}: %{x} negative items<extra></extra>"))
            fig.update_xaxes(title_text="Negative items", rangemode="tozero")
            st.plotly_chart(base_layout(fig, 320), key="topics")

    with st.expander("Show the numbers behind the charts"):
        st.dataframe(pd.crosstab(news["topic"], news["sentiment"]).reindex(columns=SENTIMENTS, fill_value=0))

    st.subheader("Needs attention")
    top = alerts[(alerts["priority"] == "High") & (alerts["status"] == "Open")].head(5)
    if top.empty:
        st.success("No open high-priority alerts.")
    for _, row in top.iterrows():
        with st.container(border=True):
            st.markdown(f"{PRIORITY_ICONS['High']} **High priority** · {row['topic']} · route to **{row['team']}**")
            st.markdown(f"**{md_safe(row['title'])}**")
            st.caption(f"{md_safe(row['source'])} · {when(row['timestamp'])} UTC · about {row['entity']} · "
                       f"{source_link(row['url'])}", unsafe_allow_html=True)

    st.subheader("Latest news")
    table = news.head(15)[["timestamp", "title", "source", "sentiment", "topic", "entity", "url"]].copy()
    table["timestamp"] = table["timestamp"].map(when)
    table["url"] = table["url"].where(table["url"].str.startswith("http"), None)    # no link -> empty cell
    st.dataframe(table, hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("Original article", display_text="Open")})


# ============================================================
# Page: News search
# ============================================================
# Page: search and browse the news, and download the results.
def page_search(news):
    """
    Page "News search": search the headlines by keyword and browse them as cards.

    The keyword search works on top of the sidebar filters. Up to 40 items are
    shown, and all matching items can be downloaded as a CSV file.

    Args:
        news: the filtered news table.
    """
    st.title("🔍 News search")
    if not news.empty and not news["url"].str.startswith("http").any():
        st.info("These headlines have no original articles to link to (demo headlines are simulated). "
                "Choose \"Live headlines (Google News)\" in the sidebar for real news with links.")
    query = st.text_input("Search the headlines", "", key="search_query",
                          placeholder="Type a keyword, for example: profit")
    if query.strip():
        news = news[news["title"].str.contains(query.strip(), case=False, na=False, regex=False)]
    st.caption(f"{len(news)} items found. The filters in the sidebar also apply.")
    if news.empty:
        st.warning("Nothing matches. Try another keyword, or widen the filters in the sidebar.")
        return
    for _, row in news.head(40).iterrows():
        news_card(row)

    export = news.drop(columns=["id"]).copy()
    st.download_button("Download these results (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="news_sentiment_results.csv", mime="text/csv")


# ============================================================
# Page: Alert centre
# ============================================================
# Page: review alerts, set their status, and edit the alert rules.
def page_alerts(alerts, all_news):
    """
    Page "Alert centre", with two tabs.

    - Alerts: list the alerts and let the user mark each one Open, Reviewed or
      Dismissed. The status is kept in st.session_state for this browser session.
    - Alert rules: change the confidence threshold and the priority topics and
      entities, or reset them to the defaults.

    Args:
        alerts:   the alerts built from the filtered news.
        all_news: the unfiltered news, used to list every entity in the rules form.
    """
    st.title("🚨 Alert centre")
    tab_list, tab_rules = st.tabs(["Alerts", "Alert rules"])

    with tab_list:
        st.caption("An alert is a news item the sentiment model classified as negative with enough confidence. "
                   "The topic model sets its priority and the team it goes to.")
        if alerts.empty:
            st.success("No alerts for the current news and rules.")
        show = st.multiselect("Show", ["Open", "Reviewed", "Dismissed"], default=["Open"], key="alert_show")
        for _, row in alerts[alerts["status"].isin(show)].iterrows():
            with st.container(border=True):
                head, action = st.columns([4, 1])
                with head:
                    st.markdown(f"{PRIORITY_ICONS[row['priority']]} **{row['priority']} priority** · {row['topic']} "
                                f"· route to **{row['team']}**")
                    st.markdown(f"**{md_safe(row['title'])}**")
                    st.caption(f"{md_safe(row['source'])} · {when(row['timestamp'])} UTC · about {row['entity']} · "
                               f"model confidence {row['confidence']:.0%} · {source_link(row['url'])}",
                               unsafe_allow_html=True)
                with action:
                    options = ["Open", "Reviewed", "Dismissed"]
                    choice = st.selectbox("Status", options, index=options.index(row["status"]),
                                          key=f"status_{row['id']}")
                    if choice != row["status"]:
                        st.session_state["alert_status"][row["id"]] = choice
                        st.rerun()

    with tab_rules:
        rules = st.session_state["rules"]
        st.caption("These rules decide which items become alerts and which of them are high priority. "
                   "They apply for this browser session.")
        with st.form("rules_form"):
            min_conf = st.slider("Raise an alert when the model is at least this confident an item is negative",
                                 0.50, 0.99, float(rules["min_confidence"]), 0.01)
            topics = st.multiselect("High priority when the topic is one of", TOPIC_NAMES,
                                    default=[t for t in rules["priority_topics"] if t in TOPIC_NAMES])
            entity_options = sorted(set(all_news["entity"]) | {COMPANY_NAME, DEMO_ENTITY})
            entities = st.multiselect("High priority when the item is about", entity_options,
                                      default=[e for e in rules["priority_entities"] if e in entity_options])
            saved = st.form_submit_button("Save rules", type="primary")
        if saved:
            st.session_state["rules"] = {"min_confidence": min_conf, "priority_topics": topics,
                                         "priority_entities": entities}
            st.rerun()
        if st.button("Reset to defaults"):
            st.session_state["rules"] = default_rules()
            st.rerun()
        st.caption("Sending alerts by email or SMS is outside this prototype.")


# ============================================================
# Page: Report
# ============================================================
# Write the summary report as Markdown text.
def build_report(news, alerts, note, for_screen=False):
    """
    Write the summary report as Markdown text.

    Sections: 1. overview counts, 2. the top 10 alerts, 3. negative items by topic,
    4. a table of sentiment by entity, followed by the names of both models.

    Args:
        news, alerts, note: the filtered news, its alerts and the source description.
        for_screen: True for the copy shown in the app (headlines are escaped so
                    Markdown displays them correctly); False for the download.
    Returns:
        The report as one Markdown string.
    """
    show = md_safe if for_screen else str        # escape only the on-screen copy
    total = len(news)
    negative = int((news["sentiment"] == "Negative").sum())
    lines = [
        f"# {COMPANY_NAME}: news sentiment report",
        "",
        f"Generated {when(utc_now())} UTC. {note}",
        "",
        "## 1. Overview",
        f"- News items analysed: {total}",
        f"- Negative: {negative} ({(negative / total if total else 0):.0%})",
        f"- Neutral: {int((news['sentiment'] == 'Neutral').sum())}",
        f"- Positive: {int((news['sentiment'] == 'Positive').sum())}",
        f"- Alerts: {len(alerts)}, of which high priority: {int((alerts['priority'] == 'High').sum())}",
        "",
        "## 2. Alerts",
    ]
    if alerts.empty:
        lines.append("No alerts.")
    for i, (_, row) in enumerate(alerts.head(10).iterrows(), 1):
        lines.append(f"{i}. **{row['priority']}** · {show(row['title'])} — {row['topic']}; route to {row['team']}; "
                     f"{show(row['source'])}, {when(row['timestamp'])} UTC")
    lines += ["", "## 3. Negative items by topic"]
    by_topic = news[news["sentiment"] == "Negative"]["topic"].value_counts()
    lines += [f"- {topic}: {count}" for topic, count in by_topic.items()] or ["None."]
    lines += ["", "## 4. Sentiment by entity", "", "| About | Negative | Neutral | Positive |", "|---|---|---|---|"]
    profile = pd.crosstab(news["entity"], news["sentiment"]).reindex(columns=SENTIMENTS, fill_value=0)
    lines += [f"| {entity} | {r['Negative']} | {r['Neutral']} | {r['Positive']} |" for entity, r in profile.iterrows()]
    lines += ["", "---",
              f"Sentiment model: {SENTIMENT_MODEL}. Topic model: {TOPIC_MODEL}. "
              "Model output is a screening aid for human review, not a verified assessment."]
    return "\n".join(lines)


# Page: preview and download the summary report.
def page_report(news, alerts, note):
    """
    Page "Report": preview the summary report and offer it as a Markdown download.

    Args:
        news, alerts, note: the filtered news, its alerts and the source description.
    """
    st.title("📄 Sentiment report")
    st.caption("A summary of the news currently selected by the sidebar filters, ready to download.")
    if news.empty:
        st.warning("Nothing to report for the current filters.")
        return
    report = build_report(news, alerts, note)
    st.download_button("Download the report (Markdown)", report.encode("utf-8"),
                       file_name=f"sentiment_report_{utc_now():%Y%m%d}.md", mime="text/markdown", type="primary")
    with st.container(border=True):
        st.markdown(build_report(news, alerts, note, for_screen=True))


# ============================================================
# Page: Analyse one text
# ============================================================
# Page: classify a single text typed by the user with both pipelines.
def page_analyse():
    """
    Page "Analyse a text": run both pipelines on a single text typed by the user.

    Shows the sentiment, topic and suggested team, says whether the text would
    raise an alert under the current rules, and shows the model's score for each class.
    """
    st.title("✍️ Analyse a piece of text")
    st.caption("Paste a headline or a sentence from a news article. English only.")
    text = st.text_area("Text", "The lender reported a sharp rise in bad loans and cut its dividend.",
                        height=110, key="one_text")
    if not st.button("Analyse", type="primary"):
        return
    if not text.strip():
        st.error("Please enter some text first.")
        return
    result = score_texts((text.strip(),)).iloc[0]
    team = TOPIC_TEAMS.get(result["topic"], DEFAULT_TEAM)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Sentiment**<br>{dot(result['sentiment'])}", unsafe_allow_html=True)
    c2.markdown(f"**Topic**<br>{result['topic']}", unsafe_allow_html=True)
    c3.markdown(f"**Suggested team**<br>{team}", unsafe_allow_html=True)

    rules = st.session_state["rules"]
    if result["sentiment"] == "Negative" and result["confidence"] >= rules["min_confidence"]:
        st.warning("This item would raise an alert under the current rules. Check the original source before acting.")
    elif result["sentiment"] == "Negative":
        st.info("Classified as negative, but below the alert threshold.")

    with st.expander("How sure is the model?"):
        st.caption("The model's own score for each class. A high score is not a guarantee that the answer is right.")
        for s, key in zip(SENTIMENTS, ["p_negative", "p_neutral", "p_positive"]):
            st.progress(float(result[key]), text=f"{s}: {result[key]:.0%}")
        st.caption(f"Topic match: {result['topic_confidence']:.0%}")


# ============================================================
# Page: Model check (accuracy of the deployed app)
# ============================================================
# Page: measure the sentiment model's accuracy on a labelled file.
def page_model_check():
    """
    Page "Model check": measure the deployed sentiment model's accuracy.

    The user uploads a CSV with 'text' and 'label' columns (or uses the bundled
    test file). Labels can be words (negative / neutral / positive) or numbers
    (0 / 1 / 2). The page shows the accuracy, a confusion table (true label vs.
    prediction), the wrongly classified items and a download of all predictions.
    At most 1,000 rows are used.
    """
    st.title("✅ Model check")
    st.caption("Measures how often the sentiment model in this app agrees with labelled examples. "
               "Upload a CSV with the columns `text` and `label` (negative, neutral or positive).")

    file = st.file_uploader("Labelled CSV", type=["csv"], key="check_file")
    use_bundled = os.path.exists(TEST_FILE) and st.checkbox(
        f"Use the test file stored with the app ({os.path.basename(TEST_FILE)})")
    if file is None and not use_bundled:
        return
    try:
        df = pd.read_csv(file if file is not None else TEST_FILE)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"text", "label"} <= set(df.columns):
            raise ValueError(f"needs the columns 'text' and 'label'; found {list(df.columns)}")
    except Exception as error:
        st.error(f"Could not use this file: {error}")
        return
    if len(df) > 1000:
        st.warning("Only the first 1,000 rows are used.")
        df = df.head(1000)

    names = {"0": "Negative", "1": "Neutral", "2": "Positive"}
    # lambda v: turn a label such as "0" or "negative" into Negative / Neutral / Positive (None if unknown)
    df["true"] = df["label"].astype(str).str.strip().str.lower().map(
        lambda v: names.get(v) or SENTIMENT_ALIASES.get(v))
    skipped = int(df["true"].isna().sum())
    df = df.dropna(subset=["true"]).reset_index(drop=True)
    if df.empty:
        st.error("No row has a label the app recognises (negative, neutral, positive, or 0, 1, 2).")
        return

    scores = score_texts(tuple(df["text"].astype(str)))
    df["predicted"] = scores["sentiment"]
    df["confidence"] = scores["confidence"]
    df["correct"] = df["true"] == df["predicted"]

    accuracy = df["correct"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("Items tested", len(df))
    c3.metric("Correct", int(df["correct"].sum()))
    if skipped:
        st.caption(f"{skipped} rows were skipped because their label was not recognised.")

    st.subheader("Where it is right and wrong")
    st.caption("Rows: the true label. Columns: what the model predicted.")
    st.dataframe(pd.crosstab(df["true"], df["predicted"]).reindex(index=SENTIMENTS, columns=SENTIMENTS, fill_value=0))

    wrong = df[~df["correct"]][["text", "true", "predicted", "confidence"]]
    with st.expander(f"Items the model got wrong ({len(wrong)})"):
        st.dataframe(wrong, hide_index=True)
    st.download_button("Download all predictions (CSV)",
                       df[["text", "true", "predicted", "confidence", "correct"]].to_csv(index=False).encode("utf-8-sig"),
                       file_name="model_check_predictions.csv", mime="text/csv")


# ============================================================
# Main
# ============================================================
# Entry point: set up the page, load the models, prepare the data and show the chosen page.
def main():
    """
    Entry point of the app. Streamlit runs this from top to bottom on every click.

    1. Set up the page and the session memory (alert rules and alert statuses).
    2. Let the user choose a page in the sidebar.
    3. Load both models (stop with a clear message if one cannot be loaded).
    4. For the news pages: load news -> run both pipelines -> filter -> build alerts,
       then show the chosen page.
    """
    st.set_page_config(page_title=f"{COMPANY_NAME} News Sentiment Monitor", page_icon="🏦", layout="wide")

    if "YOUR_HF_USERNAME" in SENTIMENT_MODEL:
        st.error("Set SENTIMENT_MODEL at the top of app.py to the model you published in notebook 02 "
                 "(for example `your_username/boc-finsentiment-distilbert`).")
        st.stop()

    st.session_state.setdefault("rules", default_rules())
    st.session_state.setdefault("alert_status", {})

    page = st.sidebar.radio(
        "Page", ["Dashboard", "News search", "Alert centre", "Report", "Analyse a text", "Model check"], key="page")

    try:
        load_sentiment_pipeline()
        load_topic_pipeline()
    except Exception as error:
        st.error(f"A model could not be loaded: {type(error).__name__}: {error}\n\n"
                 "Check the two model names at the top of app.py and that the sentiment model is public.")
        st.stop()

    if page == "Analyse a text":
        page_analyse()
    elif page == "Model check":
        page_model_check()
    else:
        raw_news, note = choose_news()                                # 1. get the headlines
        news = enrich(raw_news)                                       # 2. run both pipelines
        filtered = filter_news(news)                                  # 3. apply the sidebar filters
        alerts = build_alerts(filtered, st.session_state["rules"])    # 4. turn results into alerts
        if page == "Dashboard":
            page_dashboard(filtered, alerts, note)
        elif page == "News search":
            page_search(filtered)
        elif page == "Alert centre":
            page_alerts(alerts, news)
        else:
            page_report(filtered, alerts, note)

    st.sidebar.divider()
    st.sidebar.caption(
        f"Pipeline 1 · sentiment: [{SENTIMENT_MODEL}](https://huggingface.co/{SENTIMENT_MODEL})  \n"
        f"Pipeline 2 · topic: [{TOPIC_MODEL}](https://huggingface.co/{TOPIC_MODEL})"
    )
    st.sidebar.caption(f"Academic prototype for ISOM5240. Not affiliated with [{COMPANY_NAME}]({COMPANY_URL}). "
                       "Model output is a screening aid, not investment or compliance advice.")


if __name__ == "__main__":
    main()
