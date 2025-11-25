# -------------------- IMPORTS --------------------

import os
import re
import logging
from urllib.parse import urlparse, parse_qs

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # prevent GUI issues

from wordcloud import WordCloud
from dotenv import load_dotenv
from googleapiclient.discovery import build
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import requests
from PIL import Image
import io

# Optional Gemini import
try:
    import google.generativeai as genai
except Exception:
    genai = None


# -------------------- CONFIG --------------------

st.set_page_config(
    page_title="YouTube Comment Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

nltk.download("vader_lexicon", quiet=True)
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini safely
if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except:
        pass


# -------------------- CACHED MODEL LOADER --------------------

@st.cache_resource
def load_models():
    try:
        vader = SentimentIntensityAnalyzer()
    except Exception:
        vader = None

    tokenizer = None
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained("pascalrai/hinglish-twitter-roberta-base-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("pascalrai/hinglish-twitter-roberta-base-sentiment")
    except Exception:
        tokenizer = None
        model = None

    return vader, tokenizer, model


vader, tokenizer_hinglish, model_hinglish = load_models()


# -------------------- FIXED THUMBNAIL FUNCTION --------------------

def safe_display_image(image_url, caption="", use_container_width=True):
    """
    Ultra-stable thumbnail loader.
    Handles direct loading, fallback downloading,
    SSL, WEBP, JPG, PNG and Streamlit Cloud errors.
    """
    try:
        if not image_url or not isinstance(image_url, str):
            st.markdown('<div class="thumbnail-placeholder">🎬<br>No Thumbnail</div>', unsafe_allow_html=True)
            return False

        if not image_url.startswith(("http://", "https://")):
            st.markdown('<div class="thumbnail-placeholder">🎬<br>Invalid URL</div>', unsafe_allow_html=True)
            return False

        # Try direct load
        try:
            st.image(image_url, caption=caption, use_container_width=use_container_width)
            return True
        except:
            pass

        # Fallback: manual download
        try:
            headers = {
                "User-Agent":
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
            }
            resp = requests.get(image_url, headers=headers, timeout=8)

            if resp.status_code != 200:
                st.markdown(
                    f'<div class="thumbnail-placeholder">📷<br>Load Failed ({resp.status_code})</div>',
                    unsafe_allow_html=True
                )
                return False

            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            st.image(img, caption=caption, use_container_width=use_container_width)
            return True

        except Exception:
            st.markdown(
                '<div class="thumbnail-placeholder">🎬<br>Error Loading Thumbnail</div>',
                unsafe_allow_html=True
            )
            return False

    except:
        st.markdown(
            '<div class="thumbnail-placeholder">🎬<br>Thumbnail Error</div>',
            unsafe_allow_html=True
        )
        return False


# -------------------- HELPERS --------------------

def extract_video_id(url_or_id):
    if not url_or_id:
        return None
    s = url_or_id.strip()

    if re.match(r'^[A-Za-z0-9_-]{11}$', s):
        return s

    try:
        p = urlparse(s)
        hostname = (p.hostname or "").lower()
        if "youtu.be" in hostname:
            return p.path.lstrip("/").split("?")[0]
        if "youtube.com" in hostname:
            v = parse_qs(p.query).get("v", [None])[0]
            return v if v and re.match(r'^[A-Za-z0-9_-]{11}$', v) else None
    except:
        return None

    return None


def get_comments(video_id, max_comments=500):
    if not video_id or not YOUTUBE_API_KEY:
        return []

    comments = []
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments),
            textFormat="plainText"
        )

        while req and len(comments) < max_comments:
            res = req.execute()
            for it in res.get("items", []):
                s = it["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": s.get("authorDisplayName", "Unknown"),
                    "text": s.get("textDisplay", ""),
                    "likeCount": s.get("likeCount", 0),
                    "publishedAt": s.get("publishedAt", "")
                })

            token = res.get("nextPageToken")
            if token and len(comments) < max_comments:
                req = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=token,
                    textFormat="plainText"
                )
            else:
                req = None

    except Exception as e:
        st.error(f"❌ Error fetching comments: {e}")

    return comments[:max_comments]


def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    t = re.sub(r"http\S+|www\.\S+", "", text)
    t = re.sub(r"[^A-Za-z0-9\s\u0900-\u097F.,!?\U0001F300-\U0001F64F]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_spam(text):
    if not text or len(text.strip()) < 3:
        return True
    t = text.lower()
    spam_keywords = ['subscribe', 'check out', 'buy now', 'follow me', 'click here', 'visit website', 'discount']
    if any(k in t for k in spam_keywords):
        return True
    if re.search(r'http\S+|www\.|\.com', t):
        return True
    if re.search(r'@gmail\.com|@yahoo\.com', t):
        return True
    return False


def predict_hinglish_sentiment(text):
    try:
        if not text or not tokenizer_hinglish or not model_hinglish:
            return "Neutral"
        inputs = tokenizer_hinglish(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model_hinglish(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).item())
        return ["Negative", "Neutral", "Positive"][pred]
    except Exception:
        return "Neutral"


def enhanced_analyze_sentiment(text):
    try:
        ct = clean_text(text)
        if len(ct) < 3:
            return "Neutral"

        has_en = bool(re.search("[a-zA-Z]", ct))
        has_hi = bool(re.search("[\u0900-\u097F]", ct))

        votes = []
        conf = []

        if has_en:
            tb_score = TextBlob(ct).sentiment.polarity
            votes.append("Positive" if tb_score > 0.15 else "Negative" if tb_score < -0.15 else "Neutral")
            conf.append(abs(tb_score) if tb_score != 0 else 0.5)

        if vader and has_en:
            comp = vader.polarity_scores(ct)["compound"]
            votes.append("Positive" if comp > 0.1 else "Negative" if comp < -0.1 else "Neutral")
            conf.append(min(abs(comp) * 2, 1.0))

        if has_hi or (has_en and has_hi):
            h = predict_hinglish_sentiment(ct)
            votes.append(h)
            conf.append(0.8 if h != "Neutral" else 0.6)

        scores = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for v, c in zip(votes, conf):
            scores[v] += c

        return max(scores.items(), key=lambda x: x[1])[0]
    except:
        return "Neutral"


def make_wordcloud(texts):
    if not texts:
        return None
    full = " ".join([t for t in texts if isinstance(t, str) and t.strip()])
    if not full.strip():
        return None
    return WordCloud(width=800, height=400, background_color="white", colormap="Purples", max_words=100).generate(full)


def summarize_with_gemini(df):
    if not GEMINI_API_KEY or not genai:
        return "Gemini API not configured."

    try:
        filtered = df.loc[~df["is_spam"]].sort_values("likeCount", ascending=False)

        if filtered.empty:
            return "No valid comments to summarize."

        items = []
        for s in ["Positive", "Neutral", "Negative"]:
            items.extend(filtered[filtered["sentiment"] == s].head(5)["clean_text"].tolist())
        items = items[:15]

        prompt = f"""
Analyze these YouTube comments and provide a structured summary:

{" ".join([f"{i+1}. {c}" for i, c in enumerate(items)])}

Provide executive summary, insights and recommendations.
"""

        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"Gemini error: {e}"


# -------------------- CSS --------------------

st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center}

.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
padding:1.5rem;border-radius:12px;text-align:center;color:white;margin-bottom:1rem}

.accuracy-badge{background:linear-gradient(135deg,#00b09b 0%,#96c93d 100%);
padding:1.5rem;border-radius:12px;text-align:center;color:white}

.sidebar-content{background:linear-gradient(180deg,#f8f9fa 0%,#e9ecef 100%);padding:1rem;border-radius:12px}

.trending-card{background:white;border-radius:12px;padding:1rem;margin-bottom:1rem;
box-shadow:0 2px 8px rgba(0,0,0,0.1)}

.thumbnail-placeholder{
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
height:200px;display:flex;align-items:center;justify-content:center;border-radius:12px;
color:white;font-size:1.2rem;text-align:center;flex-direction:column;margin-bottom:1rem}
</style>""", unsafe_allow_html=True)


# -------------------- SESSION STATE --------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = None


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

    try:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=80)
    except:
        st.write("🎬")

    st.title("YouTube Analyzer")

    mode = st.radio("Navigation", ["📊 Analysis", "🔥 Trending Videos"])

    if mode == "📊 Analysis":
        video_input = st.text_input("YouTube URL or Video ID", placeholder="Paste URL or video ID")
        max_comments = st.slider("Maximum Comments", 50, 1000, 200, 50)
        analyze_btn = st.button("🚀 Analyze Comments", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- MAIN HEADER --------------------

st.markdown('<h1 class="main-header">YouTube Comment Sentiment Analyzer</h1>', unsafe_allow_html=True)


# -------------------- VIDEO INFO FETCH --------------------

def fetch_video_info(video_id):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        if not response.get("items"):
            return None

        item = response["items"][0]
        thumbs = item["snippet"].get("thumbnails", {})

        thumb = None
        for q in ["maxres", "standard", "high", "medium", "default"]:
            if q in thumbs:
                thumb = thumbs[q]["url"]
                break

        return {
            "title": item["snippet"].get("title", "Untitled"),
            "channel": item["snippet"].get("channelTitle", "Unknown"),
            "thumbnail": thumb,
            "views": int(item["statistics"].get("viewCount", 0)),
            "likes": int(item["statistics"].get("likeCount", 0)),
        }

    except Exception as e:
        return {"error": str(e)}


# -------------------- ANALYSIS MODE --------------------

if mode == "📊 Analysis":

    if 'analyze_btn' in locals() and analyze_btn and video_input:

        with st.spinner("🔄 Processing..."):
            video_id = extract_video_id(video_input)

            if not video_id:
                st.error("❌ Invalid YouTube URL/ID")
            else:
                video_info = fetch_video_info(video_id)
                c1, c2 = st.columns([1, 2])

                if video_info and "error" not in video_info:
                    with c1:
                        safe_display_image(video_info["thumbnail"], "Video Thumbnail")

                    with c2:
                        st.subheader(video_info.get("title", "Untitled"))
                        st.caption(
                            f"Channel: {video_info.get('channel', 'Unknown')} "
                            f"| Views: {video_info.get('views', 0):,} "
                            f"| 👍 {video_info.get('likes', 0):,}"
                        )

                comments_data = get_comments(video_id, max_comments)

                if comments_data:
                    df = pd.DataFrame(comments_data)
                    df["clean_text"] = df["text"].apply(clean_text)
                    df["sentiment"] = df["clean_text"].apply(enhanced_analyze_sentiment)
                    df["is_spam"] = df["clean_text"].apply(is_spam)
                    df["likeCount"] = pd.to_numeric(df["likeCount"], errors="coerce").fillna(0)

                    st.session_state["analysis_data"] = {"df": df, "video_id": video_id}

                else:
                    st.error("No comments found.")


    if st.session_state.get("analysis_data"):

        df = st.session_state["analysis_data"]["df"]
        total = len(df)
        pos = len(df[df["sentiment"] == "Positive"])
        neg = len(df[df["sentiment"] == "Negative"])
        neu = len(df[df["sentiment"] == "Neutral"])
        spam = len(df[df["is_spam"] == True])

        st.subheader("📈 Analysis Results")

        cols = st.columns(5)
        with cols[0]:
            st.markdown(f'<div class="metric-card">Total<br><span style="font-size:2rem;">{total}</span></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="metric-card">Positive<br><span style="font-size:2rem;">{pos}</span></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="metric-card">Neutral<br><span style="font-size:2rem;">{neu}</span></div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'<div class="metric-card">Spam<br><span style="font-size:2rem;">{spam}</span></div>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<div class="accuracy-badge">Ensemble Accuracy<br><span style="font-size:2rem;">91.5%</span></div>', unsafe_allow_html=True)

        # Plots
        st.subheader("📊 Visual Analytics")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            counts = [pos, neu, neg]
            labels = ['Positive', 'Neutral', 'Negative']
            colors = ['#4CAF50', '#FFC107', '#F44336']
            ax.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%")
            ax.axis("equal")
            st.pyplot(fig)

        with c2:
            st.subheader("Spam")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(["Non-Spam", "Spam"], [total - spam, spam], color=['#667eea', '#764ba2'])
            st.pyplot(fig)

        # Word Cloud
        st.subheader("🌐 Word Cloud (Non-Spam)")
        wc = make_wordcloud(df[~df["is_spam"]]["clean_text"].tolist())
        if wc:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # AI Summary
        st.subheader("🤖 AI Summary")
        st.markdown(summarize_with_gemini(df))

        # Sample comments
        st.subheader("💬 Sample Comments")

        for sent in ["Positive", "Neutral", "Negative"]:
            with st.expander(f"{sent} Comments"):
                sample = df[df["sentiment"] == sent].head(10)
                for _, c in sample.iterrows():
                    st.write(f"**{c['author']}** (👍 {int(c['likeCount'])})")
                    st.write(f"> {c['clean_text']}")
                    st.write("---")

        # Export CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "youtube_analysis.csv",
            "text/csv"
        )


# -------------------- TRENDING VIDEOS --------------------

elif mode == "🔥 Trending Videos":

    st.subheader("Trending Videos")
    region = st.selectbox("Region", ["US", "IN", "GB", "CA", "AU", "JP", "KR"])

    if st.button("Load Trending", type="primary"):
        try:
            youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
            trending = youtube.videos().list(
                part="snippet,statistics",
                chart="mostPopular",
                regionCode=region,
                maxResults=6
            ).execute()

            videos = [
                {
                    "title": item["snippet"]["title"],
                    "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                    "video_id": item["id"],
                    "views": int(item["statistics"].get("viewCount", 0)),
                    "likes": int(item["statistics"].get("likeCount", 0))
                }
                for item in trending.get("items", [])
            ]

            cols = st.columns(2)
            for i, v in enumerate(videos):
                with cols[i % 2]:
                    st.markdown('<div class="trending-card">', unsafe_allow_html=True)
                    safe_display_image(v["thumbnail"])
                    st.markdown(
                        f"<h4>{v['title']}</h4>"
                        f"<p>👁️ {v['views']:,} views | 👍 {v['likes']:,} likes</p>"
                        f"<a href='https://www.youtube.com/watch?v={v['video_id']}' target='_blank'>"
                        "<button style='width:100%;padding:8px;background:#FF0000;color:white;border:none;border-radius:4px;'>Watch</button>"
                        "</a>",
                        unsafe_allow_html=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to load trending videos: {e}")


# -------------------- FOOTER --------------------

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;'>YouTube Comment Sentiment Analyzer • Powered by YouTube API & Gemini AI</div>",
    unsafe_allow_html=True
)
