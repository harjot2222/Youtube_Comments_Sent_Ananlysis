import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from googleapiclient.discovery import build
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from groq import AuthenticationError
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
def load_hindi_bert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/hindi-bert-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("l3cube-pune/hindi-bert-sentiment")
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return classifier
    except Exception as e:
        st.error(f"❌ Failed to load Hindi BERT model: {str(e)}")
        return None

def predict_sentiment_hindi_bert(comment, classifier):
    try:
        cleaned_comment = clean_text(comment)
        if not cleaned_comment:
            return 'Neutral'
        # Truncate to 512 tokens (BERT's max length)
        result = classifier(cleaned_comment, truncation=True, max_length=512)
        label = result[0]['label']
        # hindi-bert-sentiment outputs 'positive', 'neutral', 'negative'
        return label.capitalize()
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

@st.cache_data
def summarize_comments_langchain(comments, groq_api_key):
    try:
        model = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        prompt = PromptTemplate.from_template("""
            Summarize the following YouTube comments in Hindi, English, or Hinglish. Provide a concise summary, highlight common themes, user concerns, and what users appreciated or disliked. Give actionable insights for improving future videos.

            Comments:
            {comments}

            Summary:
        """)
        chain = LLMChain(llm=model, prompt=prompt)
        response = chain.invoke({"comments": "\n".join(comments[:100])})
        return response['text'].strip()
    except AuthenticationError:
        return "❌ Summary unavailable due to invalid Groq API key."
    except Exception as e:
        return f"⚠️ Error in summarization: {str(e)}."

def qa_bot_response_langchain(user_query, comments, groq_api_key):
    try:
        model = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        prompt = PromptTemplate.from_template("""
            You are a helpful assistant analyzing YouTube comments in Hindi, English, or Hinglish. Use the following comments to answer the user's question, providing insights or suggestions for improvement where relevant:

            Comments:
            {comments}

            Question:
            {query}

            Answer:
        """)
        chain = LLMChain(llm=model, prompt=prompt)
        response = chain.invoke({"comments": "\n".join(comments[:100]), "query": user_query})
        return response['text'].strip()
    except AuthenticationError:
        return "❌ Response unavailable due to invalid Groq API key."
    except Exception as e:
        return f"⚠️ Error in response: {str(e)}."

# --------------------- Main App --------------------- #
def main():
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'comments' not in st.session_state:
        st.session_state.comments = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'bot_response' not in st.session_state:
        st.session_state.bot_response = ""

    # Load API keys from Streamlit secrets
    try:
        youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
        groq_api_key = st.secrets["GROQ_API_KEY"]
        if not groq_api_key or not re.match(r'^[a-zA-Z0-9_-]{30,100}$', groq_api_key):
            st.error("❌ Invalid Groq API key format in Streamlit secrets.")
            st.stop()
    except KeyError:
        st.error("❌ Missing API keys. Please configure YOUTUBE_API_KEY and GROQ_API_KEY in Streamlit secrets.")
        st.stop()

    # Load Hindi BERT model
    classifier = load_hindi_bert_model()
    if classifier is None:
        st.error("❌ Cannot proceed without Hindi BERT model.")
        st.stop()

    # Sidebar configuration
    with st.sidebar:
        st.image("youtube-logo-png-46020.png", use_container_width=True)
        st.header("🎬 YouTube Analyzer (Hindi & English)")
        page = st.selectbox("Navigate", ["Home", "Analysis", "CommentBot", "About"])

        if page in ["Home", "Analysis", "CommentBot"]:
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
                        lambda c: predict_sentiment_hindi_bert(c, classifier)
                    )
                    st.session_state.summary = summarize_comments_langchain(st.session_state.comments, groq_api_key)
                    st.success("✅ Analysis completed.")

    # Page rendering
    if page == "Home":
        st.title("🎯 YouTube Sentiment Analyzer (Hindi & English)")
        st.markdown("Analyze YouTube comments in Hindi, English, and Hinglish using Hindi BERT and Grok for insights!")

    elif page == "Analysis":
        st.title("📈 Analysis Results")
        if st.session_state.df is not None:
            visualize_sentiment(st.session_state.df)
            st.subheader("📝 Summary")
            st.markdown(st.session_state.summary)
        else:
            st.info("ℹ️ Run analysis from the sidebar.")

    elif page == "CommentBot":
        st.title("💬 CommentBot")
        if st.session_state.comments:
            st.session_state.user_query = st.text_input("Ask something about the comments (e.g., improvements, suggestions):", value=st.session_state.user_query)
            if st.button("Ask CommentBot"):
                with st.spinner("💡 Thinking..."):
                    st.session_state.bot_response = qa_bot_response_langchain(
                        st.session_state.user_query, st.session_state.comments, groq_api_key)
            if st.session_state.bot_response:
                st.markdown(f"**Bot Answer:** {st.session_state.bot_response}")
        else:
            st.info("⚠️ Analyze a video first to ask the bot.")

    elif page == "About":
        st.title("ℹ️ About")
        st.markdown("""
        This app analyzes YouTube comments in Hindi, English, and Hinglish with:
        - **Hindi BERT** for sentiment analysis
        - **Grok + LLaMA 3** for summarization and answering questions about comments

        Built with ❤️ using Streamlit, LangChain, and Hugging Face transformers.
        """)

    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
