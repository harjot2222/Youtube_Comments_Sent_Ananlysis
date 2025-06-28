import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from langdetect import detect, LangDetectException
import torch
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="YouTube Sentiment Analyzer (Hindi & English)", layout="wide")

# --------------------- Styling --------------------- #
st.markdown("""
<style>
body { font-family: 'Arial', sans-serif; }
.stApp { background-color: #f4f6f9; }
.stSidebar .stButton>button {
    background-color: #4CAF50; color: white; border-radius: 8px;
    width: 100%; padding: 10px; font-weight: bold;
}
.stSidebar .stButton>button:hover { background-color: #45a049; }
.main .block-container {
    padding: 2rem; background-color: #ffffff;
    border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-top: 1rem;
}
h1 { color: #1e1e2f; font-size: 2.5rem; text-align: center; margin-bottom: 1rem; }
h2 { color: #2c3e50; font-size: 1.8rem; margin-top: 1.5rem; }
.stTextInput input {
    border-radius: 8px; border: 1px solid #dcdcdc; padding: 10px;
}
.stDownloadButton>button {
    background-color: #007bff; color: white;
    border-radius: 8px; padding: 10px 20px;
}
.stDownloadButton > button:hover { background-color: #0056b3; }
</style>
""", unsafe_allow_html=True)

# --------------------- Utilities --------------------- #
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

def detect_language(comment):
    try:
        lang = detect(comment)
        # Restrict to Hindi ('hi'), English ('en'), or Hinglish (detected as 'hi' or 'en')
        return lang if lang in ['hi', 'en'] else 'hi'  # Treat Hinglish as 'hi' for simplicity
    except LangDetectException:
        return 'hi'  # Default to 'hi' for mixed or ambiguous text

@st.cache_resource
def load_muril_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
        model = AutoModelForSequenceClassification.from_pretrained("google/muril-base-cased")
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return classifier
    except Exception as e:
        st.error(f"❌ Failed to load MuRIL model: {str(e)}")
        return None

def predict_sentiment_muril(comment, classifier):
    try:
        cleaned_comment = clean_text(comment)
        if not cleaned_comment:
            return 'Neutral'
        # Truncate to 512 tokens (MuRIL's max length)
        result = classifier(cleaned_comment, truncation=True, max_length=512)
        label = result[0]['label']
        # MuRIL typically outputs LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
        if label == 'LABEL_2':
            return 'Positive'
        elif label == 'LABEL_0':
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        st.warning(f"⚠️ Sentiment analysis error for comment: {str(e)}. Defaulting to Neutral.")
        return 'Neutral'

def extract_video_id(link):
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", link)
    return match.group(1) if match else None

@st.cache_data
def get_comments(video_id, api_key, max_comments=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100,
            pageToken=next_page_token, textFormat="plainText"
        )
        try:
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            next_page_token = response.get("nextPageToken", None)
            if not next_page_token:
                break
        except Exception as e:
            st.warning(f"API Error: {str(e)}")
            break
    return comments[:max_comments]

def visualize_sentiment(results_df):
    st.subheader("📊 Sentiment Distribution")
    if 'Sentiment' in results_df.columns:
        sentiment_counts = results_df['Sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2', ax=ax)
        ax.set_ylabel("Number of Comments")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Sentiment analysis was not performed.")

    st.subheader("☁️ WordCloud of Comments")
    all_comments = " ".join(results_df["Comment"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    st.subheader("📄 Sample Comments")
    st.download_button("⬇️ Download CSV", results_df.to_csv(index=False), "yt_comments.csv", "text/csv")
    st.dataframe(results_df[['Comment', 'Sentiment', 'Language']].head(10), use_container_width=True)

# --------------------- Main App --------------------- #
def main():
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'comments' not in st.session_state:
        st.session_state.comments = None
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""

    # Load API key from Streamlit secrets
    try:
        youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
    except KeyError:
        st.error("❌ Missing YouTube API key. Please configure YOUTUBE_API_KEY in Streamlit secrets.")
        st.stop()

    # Load MuRIL model
    classifier = load_muril_model()
    if classifier is None:
        st.error("❌ Cannot proceed without MuRIL model.")
        st.stop()

    # Sidebar configuration
    with st.sidebar:
        st.image("youtube-logo-png-46020.png", use_container_width=True)
        st.header("🎬 YouTube Analyzer (Hindi & English)")
        page = st.selectbox("Navigate", ["Home", "Analysis", "About"])

        if page in ["Home", "Analysis"]:
            st.subheader("🔧 Analysis Settings")
            youtube_url = st.text_input("YouTube Video URL")
            max_comments = st.slider("Number of Comments", 50, 500, 200, step=50)

            if st.button("Analyze Now"):
                if not youtube_url:
                    st.error("❌ Please enter a valid YouTube URL.")
                    return
                with st.spinner("⏳ Analyzing comments..."):
                    video_id = extract_video_id(youtube_url)
                    if not video_id:
                        st.error("❌ Invalid YouTube URL.")
                        return
                    st.session_state.comments = get_comments(video_id, youtube_api_key, max_comments)
                    if not st.session_state.comments:
                        st.warning("⚠️ No comments fetched.")
                        return
                    # Create DataFrame with original comments
                    st.session_state.df = pd.DataFrame(st.session_state.comments, columns=["Comment"])
                    st.session_state.df['Language'] = st.session_state.df['Comment'].apply(detect_language)
                    st.session_state.df['Sentiment'] = st.session_state.df['Comment'].apply(
                        lambda c: predict_sentiment_muril(c, classifier)
                    )
                    st.success("✅ Analysis completed.")

    # Page rendering
    if page == "Home":
        st.title("🎯 YouTube Sentiment Analyzer (Hindi & English)")
        st.markdown("Analyze YouTube comments in Hindi, English, and Hinglish using Google's MuRIL model!")

    elif page == "Analysis":
        st.title("📈 Analysis Results")
        if st.session_state.df is not None:
            visualize_sentiment(st.session_state.df)
        else:
            st.info("ℹ️ Run analysis from the sidebar.")

    elif page == "About":
        st.title("ℹ️ About")
        st.markdown("""
        This app analyzes YouTube comments in Hindi, English, and Hinglish with:
        - **Google MuRIL** for sentiment analysis

        Built with ❤️ using Streamlit and Hugging Face transformers.
        """)

    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
