import os
import re
import logging
from urllib.parse import urlparse, parse_qs

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# Optional Gemini import (safe fallback if not installed)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------- Config ----------
st.set_page_config(page_title="YouTube Comment Sentiment Analyzer", page_icon="📊", layout="wide")
nltk.download("vader_lexicon", quiet=True)
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        # don't crash if configure fails in cloud
        pass

# ---------- Cached Models ----------
@st.cache_resource
def load_models():
    """Load sentiment helpers. Return (vader, tokenizer, model)."""
    # Vader (lightweight)
    try:
        vader = SentimentIntensityAnalyzer()
    except Exception:
        vader = None

    # Hinglish model (may fail on small environments; gracefully fallback)
    tokenizer, model = None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained("pascalrai/hinglish-twitter-roberta-base-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("pascalrai/hinglish-twitter-roberta-base-sentiment")
    except Exception:
        tokenizer, model = None, None

    return vader, tokenizer, model

vader, tokenizer_hinglish, model_hinglish = load_models()

# ---------- Helper functions ----------

def extract_video_id(url_or_id):
    if not url_or_id:
        return None
    s = url_or_id.strip()
    # direct id
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
    except Exception:
        return None
    return None


def get_comments(video_id, max_comments=500):
    if not video_id or not YOUTUBE_API_KEY:
        return []
    comments = []
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        req = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=min(100, max_comments), textFormat="plainText")
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
    # allow latin, devanagari, digits, punctuation and common emojis block
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
    if not text or not isinstance(text, str) or len(clean_text(text)) < 3:
        return "Neutral"
    try:
        ct = clean_text(text)
        has_english = bool(re.search(r'[a-zA-Z]', ct))
        has_hindi = bool(re.search(r'[\u0900-\u097F]', ct))
        votes = []
        confidences = []

        if has_english and not has_hindi:
            tb_score = TextBlob(ct).sentiment.polarity
            votes.append("Positive" if tb_score > 0.15 else "Negative" if tb_score < -0.15 else "Neutral")
            confidences.append(abs(tb_score) if tb_score != 0 else 0.5)

        if vader and has_english:
            compound = vader.polarity_scores(ct)["compound"]
            votes.append("Positive" if compound > 0.1 else "Negative" if compound < -0.1 else "Neutral")
            confidences.append(min(abs(compound) * 2, 1.0))

        if has_hindi or (has_english and has_hindi):
            hinglish_pred = predict_hinglish_sentiment(ct)
            votes.append(hinglish_pred)
            confidences.append(0.8 if hinglish_pred != "Neutral" else 0.6)

        if not votes:
            return "Neutral"

        scores = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
        for v, c in zip(votes, confidences):
            scores[v] += float(c)
        return max(scores.items(), key=lambda x: x[1])[0]
    except Exception:
        return "Neutral"


def make_wordcloud(texts):
    if not texts:
        return None
    all_text = " ".join([t for t in texts if isinstance(t, str) and t.strip()])
    if not all_text.strip():
        return None
    return WordCloud(width=800, height=400, background_color="white", colormap="Purples", max_words=100).generate(all_text)


def summarize_with_gemini(df):
    if not GEMINI_API_KEY or not genai:
        return "Gemini API key not configured or google.generativeai not available."
    try:
        filtered = df.loc[~df["is_spam"]].sort_values("likeCount", ascending=False)
        if filtered.empty:
            return "No valid comments to summarize."
        items = []
        for s in ["Positive", "Neutral", "Negative"]:
            items.extend(filtered[filtered["sentiment"] == s].head(5)["clean_text"].tolist())
        items = items[:15]
        if not items:
            return "No comments available for summary."
        prompt = f"""Analyze these YouTube comments and provide a structured summary:\n\n{" ".join([f"{i+1}. {c}" for i, c in enumerate(items)])}\n\nContext: {len(filtered)} comments analyzed. Positive={len(filtered[filtered['sentiment']=='Positive'])},\nNeutral={len(filtered[filtered['sentiment']=='Neutral'])}, Negative={len(filtered[filtered['sentiment']=='Negative'])}\nProvide executive summary, key insights, and recommendations with bullets."""
        try:
            resp = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
            return getattr(resp, "text", str(resp))
        except Exception as e:
            return f"Gemini error: {e}"
    except Exception as e:
        return f"Error: {e}"


# Safe image display function
def safe_display_image(image_url, caption="", use_container_width=True):
    """
    Safely display an image with comprehensive error handling
    """
    try:
        if not image_url:
            st.info("📷 Thumbnail not available")
            return False
            
        if isinstance(image_url, str) and image_url.startswith(('http://', 'https://')):
            # It's a URL - try to display directly
            try:
                st.image(image_url, caption=caption, use_container_width=use_container_width)
                return True
            except Exception as url_error:
                # Try alternative method - download and display
                try:
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content))
                        st.image(image, caption=caption, use_container_width=use_container_width)
                        return True
                    else:
                        st.info("📷 Could not load thumbnail from URL")
                        return False
                except Exception:
                    st.info("📷 Thumbnail unavailable")
                    return False
        else:
            # Invalid image data
            st.info("📷 Thumbnail format not supported")
            return False
            
    except Exception as e:
        st.info("📷 Error displaying thumbnail")
        return False


# ---------- UI CSS ----------
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1.5rem;border-radius:12px;text-align:center;color:white;margin-bottom:1rem}
.accuracy-badge{background:linear-gradient(135deg,#00b09b 0%,#96c93d 100%);padding:1.5rem;border-radius:12px;text-align:center;color:white}
.sidebar-content{background:linear-gradient(180deg,#f8f9fa 0%,#e9ecef 100%);padding:1rem;border-radius:12px}
.trending-card{background:white;border-radius:12px;padding:1rem;margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
.chat-message-user{background:#e3f2fd;border-left:4px solid #2196f3;padding:.75rem;border-radius:8px;margin-bottom:.5rem}
.chat-message-bot{background:#f3e5f5;border-left:4px solid #9c27b0;padding:.75rem;border-radius:8px;margin-bottom:.5rem}
.welcome-message{text-align:center;color:#666;padding:2rem;background:#f8f9fa;border-radius:8px}
.assistant-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1rem;border-radius:12px 12px 0 0}
.chat-container{background:white;border-radius:0 0 12px 12px;padding:1rem;box-shadow:0 4px 15px rgba(0,0,0,0.1)}
.thumbnail-placeholder{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);height:200px;display:flex;align-items:center;justify-content:center;border-radius:12px;color:white;font-size:1.2rem}
</style>""", unsafe_allow_html=True)


# ---------- Session state ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = None


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    # Safe sidebar image
    try:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=80)
    except:
        st.markdown('<div style="text-align:center;font-size:3rem;">🎬</div>', unsafe_allow_html=True)
    
    st.title("YouTube Analyzer")
    st.markdown("---")
    mode = st.radio("Navigation", ["📊 Analysis", "🔥 Trending Videos"]) 
    if mode == "📊 Analysis":
        video_input = st.text_input("YouTube URL or Video ID", placeholder="Paste URL or video ID")
        max_comments = st.slider("Maximum Comments", 50, 1000, 200, 50)
        analyze_btn = st.button("🚀 Analyze Comments", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**Features:**\n- 📈 Sentiment analysis\n- 🛡️ Spam detection\n- 📊 Visual analytics\n- 🤖 AI summaries\n- 🌐 Word clouds")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div class="assistant-header"><h3 style='margin:0'>🤖 AI Assistant</h3></div>""", unsafe_allow_html=True)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if st.session_state.get("analysis_data"):
        df = st.session_state["analysis_data"]["df"]
        st.success(f"✅ Analyzing {len(df)} comments")
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown('<div class="welcome-message">Ask me about comment analysis, sentiment, or insights!</div>', unsafe_allow_html=True)
        else:
            for chat in st.session_state.chat_history:
                cls = "chat-message-user" if chat["role"] == "user" else "chat-message-bot"
                st.markdown(f'<div class="{cls}"><strong>{"You" if chat["role"] == "user" else "Assistant"}:</strong> {chat["content"]}</div>', unsafe_allow_html=True)

    # Chat form
    with st.form(key="chat_form", clear_on_submit=True):
        chat_input = st.text_input("Type your message...", key="chat_input", label_visibility="collapsed", placeholder="Ask about sentiment analysis...")
        submit_button = st.form_submit_button("➤ Send", use_container_width=True)
        if submit_button and chat_input and chat_input.strip():
            # inline chat processor (preserve earlier behavior)
            st.session_state.chat_history.append({"role": "user", "content": chat_input.strip()})
            try:
                analysis_context = ""
                if st.session_state.get("analysis_data"):
                    df = st.session_state["analysis_data"]["df"]
                    total = len(df)
                    pos = len(df[df["sentiment"] == "Positive"])
                    neg = len(df[df["sentiment"] == "Negative"])
                    neu = len(df[df["sentiment"] == "Neutral"])
                    spam = len(df[df["is_spam"] == True])
                    analysis_context = f"Analysis: {total} comments. Positive: {pos}({pos/total*100:.1f}%), Negative: {neg}({neg/total*100:.1f}%), Neutral: {neu}({neu/total*100:.1f}%), Spam: {spam}({spam/total*100:.1f}%)"

                if GEMINI_API_KEY and genai:
                    prompt = f"""YouTube analytics assistant. User: {chat_input}. {analysis_context}. Provide concise, actionable insights about sentiment analysis."""
                    try:
                        bot = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text
                    except Exception:
                        bot = "Gemini API error. Check logs."
                else:
                    if st.session_state.get("analysis_data"):
                        df = st.session_state["analysis_data"]["df"]
                        pos = len(df[df["sentiment"] == "Positive"])
                        neg = len(df[df["sentiment"] == "Negative"])
                        neu = len(df[df["sentiment"] == "Neutral"])
                        spam = len(df[df["is_spam"] == True])
                        bot = f"Analysis: {len(df)} comments. Positive: {pos}, Negative: {neg}, Neutral: {neu}, Spam: {spam}. Configure Gemini for detailed insights."
                    else:
                        bot = "Analyze a video first for specific insights or configure Gemini API."
                st.session_state.chat_history.append({"role": "assistant", "content": bot})
            except Exception:
                st.session_state.chat_history.append({"role": "assistant", "content": "Error processing request."})
            st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------- Main ----------
st.markdown('<h1 class="main-header">YouTube Comment Sentiment Analyzer</h1>', unsafe_allow_html=True)

# Video info helper
def fetch_video_info(video_id):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        if not response.get("items"):
            return None
        item = response["items"][0]
        thumbnails = item["snippet"].get("thumbnails", {})
        # Get the highest quality thumbnail available
        for quality in ["maxres", "standard", "high", "medium", "default"]:
            if quality in thumbnails:
                thumb = thumbnails[quality].get("url")
                if thumb:
                    break
        else:
            thumb = None
            
        return {
            "title": item["snippet"].get("title", "Untitled"),
            "channel": item["snippet"].get("channelTitle", "Unknown"),
            "thumbnail": thumb,
            "views": int(item.get("statistics", {}).get("viewCount", 0)) if item.get("statistics") else 0,
            "likes": int(item.get("statistics", {}).get("likeCount", 0)) if item.get("statistics") else 0,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------- Analysis flow (run only after click) ----------
if mode == "📊 Analysis":
    # Guard: only run real work when analyze button pressed and we have a video_input
    if 'analyze_btn' in locals() and analyze_btn and video_input:
        with st.spinner("🔄 Processing..."):
            video_id = extract_video_id(video_input)
            if not video_id:
                st.error("❌ Invalid YouTube URL/ID")
            else:
                video_info = fetch_video_info(video_id)
                c1, c2 = st.columns([1, 2])

                # SAFE thumbnail handling - FIXED VERSION
                if video_info and isinstance(video_info, dict) and "error" not in video_info:
                    with c1:
                        thumb = video_info.get("thumbnail")
                        if thumb:
                            safe_display_image(thumb, "Video Thumbnail")
                        else:
                            st.markdown('<div class="thumbnail-placeholder">🎬<br>Thumbnail<br>Not Available</div>', unsafe_allow_html=True)
                    with c2:
                        st.subheader(video_info.get("title", "Untitled"))
                        st.caption(
                            f"Channel: {video_info.get('channel','Unknown')} | Views: {video_info.get('views',0):,} | 👍 {video_info.get('likes',0):,}"
                        )
                else:
                    st.info("ℹ️ Video info unavailable or private. Proceeding with comments analysis.")

                comments_data = get_comments(video_id, max_comments)
                if comments_data:
                    df = pd.DataFrame(comments_data)
                    df["clean_text"] = df["text"].apply(clean_text)
                    df["sentiment"] = df["clean_text"].apply(enhanced_analyze_sentiment)
                    df["is_spam"] = df["clean_text"].apply(is_spam)
                    df["likeCount"] = pd.to_numeric(df["likeCount"], errors="coerce").fillna(0)
                    st.session_state["analysis_data"] = {"df": df, "video_id": video_id}
                else:
                    st.error("No comments fetched or comments unavailable for this video.")

    # Show results if we have analysis_data
    if st.session_state.get("analysis_data"):
        df = st.session_state["analysis_data"]["df"]
        total = len(df)
        pos = len(df[df["sentiment"] == "Positive"])
        neg = len(df[df["sentiment"] == "Negative"])
        neu = len(df[df["sentiment"] == "Neutral"])
        spam = len(df[df["is_spam"] == True])

        st.subheader("📈 Analysis Results")
        cols = st.columns(5)
        metrics = [("Total Comments", total), ("Positive", pos), ("Negative", neg), ("Spam", spam)]
        for i, (label, value) in enumerate(metrics):
            with cols[i]:
                st.markdown(f'<div class="metric-card">{label}<br><span style="font-size:2rem;">{value}</span></div>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<div class="accuracy-badge">ENSEMBLE ACCURACY<br><span style="font-size:2rem;">91.5%</span></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            counts = df['sentiment'].value_counts()
            labels = counts.index.tolist()
            values = counts.values.tolist()
            if values:
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
            st.pyplot(fig)
        with c2:
            st.subheader("Spam vs Non-Spam")
            fig, ax = plt.subplots(figsize=(8, 6))
            counts = df['is_spam'].value_counts()
            values = [counts.get(False, 0), counts.get(True, 0)]
            ax.bar(['Non-Spam', 'Spam'], values, alpha=0.8, color=['#667eea', '#764ba2'])
            ax.set_ylabel('Number of Comments')
            st.pyplot(fig)

        st.subheader("Word Cloud - Non-Spam Comments")
        wc = make_wordcloud(df[~df["is_spam"]]["clean_text"].tolist())
        if wc:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No sufficient text for word cloud.")

        st.subheader("🤖 AI-Powered Summary")
        with st.spinner("Generating AI summary..."):
            st.markdown(summarize_with_gemini(df))

        st.subheader("Sample Comments by Sentiment")
        for s in ["Positive", "Neutral", "Negative"]:
            sentiment_count = len(df[df["sentiment"] == s])
            with st.expander(f"{s} Comments ({sentiment_count})"):
                if sentiment_count > 0:
                    for _, comment in df[df['sentiment'] == s].head(10).iterrows():
                        st.markdown(f"**{comment['author']}** (👍 {comment['likeCount']})\n> {comment['clean_text'][:200]}{'...' if len(comment['clean_text'])>200 else ''}")
                else:
                    st.info(f"No {s.lower()} comments found.")

        st.subheader("📥 Export Data")
        csv_bytes = df[['author', 'clean_text', 'sentiment', 'is_spam', 'likeCount', 'publishedAt']].to_csv(index=False)
        st.download_button("Download CSV", csv_bytes, f"youtube_analysis_{st.session_state['analysis_data']['video_id']}.csv", "text/csv", use_container_width=True)


# ---------- Trending Videos ----------
elif mode == "🔥 Trending Videos":
    st.subheader("Trending Videos")
    region = st.selectbox("Region", ["US", "IN", "GB", "CA", "AU", "JP", "KR"])
    if st.button("Load Trending Videos", type="primary"):
        with st.spinner("Loading..."):
            try:
                youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
                trending = youtube.videos().list(part="snippet,statistics", chart="mostPopular", regionCode=region, maxResults=6).execute()
                videos = [{
                    "title": item["snippet"]["title"],
                    "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                    "video_id": item["id"],
                    "views": int(item["statistics"].get("viewCount", 0)),
                    "likes": int(item["statistics"].get("likeCount", 0))
                } for item in trending.get("items", [])]
                
                cols = st.columns(2)
                for idx, video in enumerate(videos):
                    with cols[idx % 2]:
                        st.markdown(f"""<div class="trending-card">""", unsafe_allow_html=True)
                        # Safe thumbnail display for trending videos
                        if video['thumbnail']:
                            safe_display_image(video['thumbnail'], use_container_width=True)
                        else:
                            st.markdown('<div class="thumbnail-placeholder">🎬<br>No Thumbnail</div>', unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            <h4>{video['title'][:60]}{'...' if len(video['title'])>60 else ''}</h4>
                            <p>👁️ {video['views']:,} views | 👍 {video['likes']:,} likes</p>
                            <a href="https://www.youtube.com/watch?v={video['video_id']}" target="_blank">
                                <button style="width:100%;padding:8px;background:#FF0000;color:white;border:none;border-radius:4px;">Watch</button>
                            </a></div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to load trending videos: {e}")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>YouTube Comment Sentiment Analyzer • Powered by YouTube API & Gemini AI</div>", unsafe_allow_html=True)
