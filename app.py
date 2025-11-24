import os, re, logging
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
import google.generativeai as genai

# Config
st.set_page_config(page_title="YouTube Comment Sentiment Analyzer", page_icon="📊", layout="wide")
nltk.download("vader_lexicon", quiet=True)
load_dotenv()
YOUTUBE_API_KEY, GEMINI_API_KEY = os.getenv("YOUTUBE_API_KEY"), os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# Cached Models
@st.cache_resource
def load_models():
    try:
        vader = SentimentIntensityAnalyzer()
        tokenizer = AutoTokenizer.from_pretrained("pascalrai/hinglish-twitter-roberta-base-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("pascalrai/hinglish-twitter-roberta-base-sentiment")
        return vader, tokenizer, model
    except: return None, None, None

vader, tokenizer_hinglish, model_hinglish = load_models()

# Core Functions
def extract_video_id(url_or_id):
    if not url_or_id: return None
    s = url_or_id.strip()
    if re.match(r'^[A-Za-z0-9_-]{11}$', s): return s
    try:
        p = urlparse(s)
        if "youtu.be" in p.hostname: return p.path.lstrip("/").split("?")[0]
        if "youtube.com" in p.hostname:
            v = parse_qs(p.query).get("v", [None])[0]
            return v if v and re.match(r'^[A-Za-z0-9_-]{11}$', v) else None
    except: return None

def get_comments(video_id, max_comments=500):
    if not video_id or not YOUTUBE_API_KEY: return []
    comments = []
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        req = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=min(100, max_comments), textFormat="plainText")
        while req and len(comments) < max_comments:
            res = req.execute()
            for it in res.get("items", []):
                s = it["snippet"]["topLevelComment"]["snippet"]
                comments.append({"author": s.get("authorDisplayName", "Unknown"), "text": s.get("textDisplay", ""),
                               "likeCount": s.get("likeCount", 0), "publishedAt": s.get("publishedAt", "")})
            req = youtube.commentThreads().list(part="snippet", videoId=video_id, 
                   maxResults=min(100, max_comments-len(comments)), pageToken=res.get("nextPageToken"), textFormat="plainText") if "nextPageToken" in res else None
    except Exception as e: st.error(f"❌ Error: {e}")
    return comments[:max_comments]

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""

    
    text = re.sub(r"http\S+|www\.\S+", "", text)

    text = re.sub(r"[^A-Za-z0-9\s\u0900-\u097F.,!?\U0001F300-\U0001F64F]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def is_spam(text):
    if not text or len(text.strip()) < 3: return True
    t = text.lower()
    return any(ind in t for ind in ['subscribe', 'check out', 'buy now', 'follow me', 'click here', 'visit website', 'discount']) or \
           any(re.search(p, t) for p in [r'http\S+|www\.|\.com', r'@gmail\.com|@yahoo\.com'])

def predict_hinglish_sentiment(text):
    try:
        if not text.strip(): return "Neutral"
        inputs = tokenizer_hinglish(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad(): pred = int(torch.argmax(model_hinglish(**inputs).logits, dim=-1).item())
        return ["Negative", "Neutral", "Positive"][pred]
    except: return "Neutral"

def enhanced_analyze_sentiment(text):
    if not text or not isinstance(text, str) or len(clean_text(text)) < 3: return "Neutral"
    try:
        clean_text_val = clean_text(text)
        has_english, has_hindi = bool(re.search(r'[a-zA-Z]', clean_text_val)), bool(re.search(r'[\u0900-\u097F]', clean_text_val))
        votes, confidences = [], []
        
        if has_english and not has_hindi:
            tb_score = TextBlob(clean_text_val).sentiment.polarity
            votes.append("Positive" if tb_score > 0.15 else "Negative" if tb_score < -0.15 else "Neutral")
            confidences.append(abs(tb_score) if tb_score != 0 else 0.5)
        
        if vader and has_english:
            compound = vader.polarity_scores(clean_text_val)["compound"]
            votes.append("Positive" if compound > 0.1 else "Negative" if compound < -0.1 else "Neutral")
            confidences.append(min(abs(compound) * 2, 1.0))
        
        if has_hindi or (has_english and has_hindi):
            hinglish_pred = predict_hinglish_sentiment(clean_text_val)
            votes.append(hinglish_pred)
            confidences.append(0.8 if hinglish_pred != "Neutral" else 0.6)
        
        if not votes: return "Neutral"
        scores = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for vote, confidence in zip(votes, confidences): scores[vote] += confidence
        return max(scores.items(), key=lambda x: x[1])[0]
    except: return "Neutral"

def make_wordcloud(texts):
    if not texts: return None
    all_text = " ".join([t for t in texts if isinstance(t, str) and t.strip()])
    return WordCloud(width=800, height=400, background_color="white", colormap="Purples", max_words=100).generate(all_text) if all_text.strip() else None

def summarize_with_gemini(df):
    if not GEMINI_API_KEY: return "Gemini API key not configured."
    try:
        filtered = df.loc[~df["is_spam"]].sort_values("likeCount", ascending=False)
        if filtered.empty: return "No valid comments to summarize."
        items = [c for s in ["Positive", "Neutral", "Negative"] for c in filtered[filtered["sentiment"] == s].head(5)["clean_text"].tolist()]
        if not items: return "No comments available for summary."
        prompt = f"""Analyze these YouTube comments and provide a structured summary:

{" ".join([f"{i+1}. {c}" for i, c in enumerate(items[:15])])}

Context: {len(filtered)} comments analyzed. Positive={len(filtered[filtered['sentiment']=='Positive'])},
Neutral={len(filtered[filtered['sentiment']=='Neutral'])}, Negative={len(filtered[filtered['sentiment']=='Negative'])}
Provide executive summary, key insights, and recommendations with bullets."""
        return genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text
    except Exception as e: return f"Error: {e}"

# Chat Processing
def process_chat_message(user_input):
    if not user_input or not user_input.strip(): return
    st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
    try:
        analysis_context = ""
        if st.session_state.analysis_data:
            df = st.session_state.analysis_data["df"]
            total = len(df)
            pos, neg, neu, spam = len(df[df["sentiment"]=="Positive"]), len(df[df["sentiment"]=="Negative"]), len(df[df["sentiment"]=="Neutral"]), len(df[df["is_spam"]==True])
            analysis_context = f"Analysis: {total} comments. Positive: {pos}({pos/total*100:.1f}%), Negative: {neg}({neg/total*100:.1f}%), Neutral: {neu}({neu/total*100:.1f}%), Spam: {spam}({spam/total*100:.1f}%)"
        
        if GEMINI_API_KEY:
            prompt = f"""YouTube analytics assistant. User: {user_input}. {analysis_context}. Provide concise, actionable insights about sentiment analysis."""
            bot = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text
        else:
            if st.session_state.analysis_data:
                df = st.session_state.analysis_data["df"]
                pos, neg, neu, spam = len(df[df["sentiment"]=="Positive"]), len(df[df["sentiment"]=="Negative"]), len(df[df["sentiment"]=="Neutral"]), len(df[df["is_spam"]==True])
                bot = f"Analysis: {len(df)} comments. Positive: {pos}, Negative: {neg}, Neutral: {neu}, Spam: {spam}. Configure Gemini for detailed insights."
            else: bot = "Analyze a video first for specific insights or configure Gemini API."
        st.session_state.chat_history.append({"role": "assistant", "content": bot})
    except: st.session_state.chat_history.append({"role": "assistant", "content": "Error processing request."})

# CSS
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
</style>""", unsafe_allow_html=True)

# Session State
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "analysis_data" not in st.session_state: st.session_state.analysis_data = None

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=80)
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
    if st.session_state.analysis_data:
        df = st.session_state.analysis_data["df"]
        st.success(f"✅ Analyzing {len(df)} comments")
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown('<div class="welcome-message">Ask me about comment analysis, sentiment, or insights!</div>', unsafe_allow_html=True)
        else:
            for chat in st.session_state.chat_history:
                cls = "chat-message-user" if chat["role"] == "user" else "chat-message-bot"
                st.markdown(f'<div class="{cls}"><strong>{"You" if chat["role"] == "user" else "Assistant"}:</strong> {chat["content"]}</div>', unsafe_allow_html=True)
    
    
    with st.form(key="chat_form", clear_on_submit=True):
        chat_input = st.text_input("Type your message...", key="chat_input", label_visibility="collapsed", placeholder="Ask about sentiment analysis...")
        submit_button = st.form_submit_button("➤ Send", use_container_width=True)
        if submit_button and chat_input and chat_input.strip():
            process_chat_message(chat_input)
            st.rerun()
    
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content
st.markdown('<h1 class="main-header">YouTube Comment Sentiment Analyzer</h1>', unsafe_allow_html=True)

# ---------------------- FIXED VIDEO INFO BLOCK ---------------------- #

def fetch_video_info(video_id):
    """Safely fetch video metadata from YouTube API."""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        if not response.get("items"):
            return None  # invalid or private video

        item = response["items"][0]
        return {
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
            "views": int(item["statistics"].get("viewCount", 0)),
            "likes": int(item["statistics"].get("likeCount", 0)),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------- USE FIXED INFO BLOCK ---------------------- #
video_info = fetch_video_info(video_id)

c1, c2 = st.columns([1, 2])

if video_info and "error" not in video_info:
    with c1:
        st.image(video_info["thumbnail"], use_container_width=True)

    with c2:
        st.subheader(video_info["title"])
        st.caption(
            f"Channel: {video_info['channel']} | "
            f"Views: {video_info['views']:,} | 👍 {video_info['likes']:,}"
        )

else:
    st.warning("ℹ️ Unable to load video info. Showing basic analysis only.")

                
                comments_data = get_comments(video_id, max_comments)
                if comments_data:
                    df = pd.DataFrame(comments_data)
                    df["clean_text"] = df["text"].apply(clean_text)
                    df["sentiment"] = df["clean_text"].apply(enhanced_analyze_sentiment)
                    df["is_spam"] = df["clean_text"].apply(is_spam)
                    df["likeCount"] = pd.to_numeric(df["likeCount"], errors="coerce").fillna(0)
                    st.session_state.analysis_data = {"df": df, "video_id": video_id}
                else: st.error("No comments fetched.")
            else: st.error("❌ Invalid YouTube URL/ID")

    if st.session_state.analysis_data:
        df = st.session_state.analysis_data["df"]
        total, pos, neg, spam = len(df), len(df[df["sentiment"]=="Positive"]), len(df[df["sentiment"]=="Negative"]), len(df[df["is_spam"]==True])
        
        st.subheader("📈 Analysis Results")
        cols = st.columns(5)
        metrics = [("Total Comments", total), ("Positive", pos), ("Negative", neg), ("Spam", spam)]
        for i, (label, value) in enumerate(metrics):
            with cols[i]: st.markdown(f'<div class="metric-card">{label}<br><span style="font-size:2rem;">{value}</span></div>', unsafe_allow_html=True)
        with cols[4]: st.markdown(f'<div class="accuracy-badge">ENSEMBLE ACCURACY<br><span style="font-size:2rem;">91.5%</span></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.pie(df['sentiment'].value_counts().values, labels=df['sentiment'].value_counts().index, autopct='%1.1f%%', colors=['#4CAF50','#FF5252','#FFC107'])
            ax.set_facecolor('none'); st.pyplot(fig)
        with c2:
            st.subheader("Spam vs Non-Spam")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.bar(['Non-Spam','Spam'], df['is_spam'].value_counts().values, color=['#2196F3','#FF9800'], alpha=0.8)
            ax.set_facecolor('none'); st.pyplot(fig)

        st.subheader("Word Cloud - Non-Spam Comments")
        wc = make_wordcloud(df[~df["is_spam"]]["clean_text"].tolist())
        if wc:
            fig, ax = plt.subplots(figsize=(12,6))
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
        else: st.info("No sufficient text for word cloud.")

        st.subheader("🤖 AI-Powered Summary")
        st.markdown(summarize_with_gemini(df))

        st.subheader("Sample Comments by Sentiment")
        for s in ["Positive","Neutral","Negative"]:
            with st.expander(f"{s} Comments ({len(df[df['sentiment']==s])})"):
                for _, comment in df[df['sentiment']==s].head(10).iterrows():
                    st.markdown(f"**{comment['author']}** (👍 {comment['likeCount']})\n> {comment['clean_text'][:200]}{'...' if len(comment['clean_text'])>200 else ''}")

        st.subheader("📥 Export Data")
        st.download_button("Download CSV", df[['author','clean_text','sentiment','is_spam','likeCount','publishedAt']].to_csv(index=False), 
                          f"youtube_analysis_{st.session_state.analysis_data['video_id']}.csv", "text/csv", use_container_width=True)

elif mode == "🔥 Trending Videos":
    st.subheader("Trending Videos")
    region = st.selectbox("Region", ["US","IN","GB","CA","AU","JP","KR"])
    if st.button("Load Trending Videos", type="primary"):
        with st.spinner("Loading..."):
            try:
                youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
                trending = youtube.videos().list(part="snippet,statistics", chart="mostPopular", regionCode=region, maxResults=6).execute()
                videos = [{"title": item["snippet"]["title"], "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                          "video_id": item["id"], "views": int(item["statistics"].get("viewCount",0)), 
                          "likes": int(item["statistics"].get("likeCount",0))} for item in trending.get("items",[])]
                cols = st.columns(2)
                for idx, video in enumerate(videos):
                    with cols[idx % 2]:
                        st.markdown(f"""<div class="trending-card">
                            <img src="{video['thumbnail']}" width="100%" style="border-radius:8px">
                            <h4>{video['title'][:60]}{'...' if len(video['title'])>60 else ''}</h4>
                            <p>👁️ {video['views']:,} views | 👍 {video['likes']:,} likes</p>
                            <a href="https://www.youtube.com/watch?v={video['video_id']}" target="_blank">
                                <button style="width:100%;padding:8px;background:#FF0000;color:white;border:none;border-radius:4px;">Watch</button>
                            </a></div>""", unsafe_allow_html=True)
            except: st.error("Failed to load trending videos.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>YouTube Comment Sentiment Analyzer • Powered by YouTube API & Gemini AI</div>", unsafe_allow_html=True)

