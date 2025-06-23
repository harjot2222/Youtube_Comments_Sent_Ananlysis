import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")

def load_xlm_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        cache_dir="./model_cache",  # optional: to avoid permission issues
        force_download=True         # force re-download if broken
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        cache_dir="./model_cache",
        force_download=True
    )
    return tokenizer, model

tokenizer, xlm_model = load_xlm_sentiment_model()

st.markdown("""
<style>
body { font-family: 'Arial', sans-serif; }
.stApp { background-color: #f4f6f9; }
.css-1d391kg { background-color: #1e1e2f; color: #ffffff; }
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
.stDownloadButton>button:hover { background-color: #0056b3; }
</style>
""", unsafe_allow_html=True)

def clean_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

def predict_sentiment(comment):
    cleaned = clean_text(comment)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True)
    outputs = xlm_model(**inputs)
    probs = softmax(outputs.logits.detach().numpy()[0])
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    return sentiment_labels[np.argmax(probs)]

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
            if "nextPageToken" in response:
                next_page_token = response["nextPageToken"]
            else:
                break
        except Exception as e:
            st.warning(f"API limit reached or error: {str(e)}")
            break
    return comments[:max_comments]

def visualize_sentiment(results_df):
    if results_df is None or 'Sentiment' not in results_df.columns:
        st.error("No sentiment data available. Please run the analysis first.")
        return
    with st.container():
        st.subheader("\U0001F4CA Sentiment Distribution")
        sns.set_style("whitegrid")
        sentiment_counts = results_df['Sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2', ax=ax)
        ax.set_ylabel("Number of Comments")
        ax.set_title("Sentiment Analysis Results")
        st.pyplot(fig)

        st.subheader("\u2601\ufe0f WordCloud of Comments")
        all_comments = " ".join(results_df["Comment"])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

@st.cache_data
def summarize_comments_langchain(comments, groq_api_key):
    model = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    prompt = PromptTemplate.from_template(
        """
        Summarize the following YouTube comments. Provide a short summary, highlight common themes, user concerns, and what users appreciated or disliked. Give actionable insights for improving future videos.

        Comments:
        {comments}
        """
    )
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.invoke({"comments": "\n".join(comments[:100])})
    return response['text'].strip()

def qa_bot_response_langchain(user_query, comments, groq_api_key):
    model = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant analyzing YouTube comments. Use the following comments to answer the user's question:

        Comments:
        {comments}

        Question:
        {query}

        Answer:
        """
    )
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.invoke({"comments": "\n".join(comments[:100]), "query": user_query})
    return response['text'].strip()

def main():
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

    try:
        youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
        grooq_api_key = st.secrets["GROOQ_API_KEY"]
    except KeyError:
        st.error("Missing API keys. Please configure YOUTUBE_API_KEY and GROOQ_API_KEY in Streamlit secrets.")
        st.stop()

    with st.sidebar:
        st.image("youtube-logo-png-46020.png", use_container_width=True)
        st.header("\U0001F3AC YouTube Analyzer")
        page = st.selectbox("Navigate", ["Home", "Analysis", "CommentBot", "About"])

        if page in ["Home", "Analysis", "CommentBot"]:
            st.subheader("\U0001F511 Analysis Settings")
            youtube_url = st.text_input("YouTube Video URL")
            max_comments = st.slider("Number of Comments to Analyze", min_value=50, max_value=500, value=200, step=50)
            if st.button("Analyze Now", key="analyze_button"):
                if not youtube_url:
                    st.error("Please provide a valid YouTube URL.")
                    return
                with st.spinner("\u23F3 Fetching and Analyzing..."):
                    video_id = extract_video_id(youtube_url)
                    if not video_id:
                        st.error("Invalid YouTube URL.")
                        return
                    st.session_state.comments = get_comments(video_id, youtube_api_key, max_comments)
                    if not st.session_state.comments:
                        st.warning("No comments fetched.")
                        return
                    st.session_state.df = pd.DataFrame(st.session_state.comments, columns=["Comment"])
                    st.session_state.df['Sentiment'] = st.session_state.df['Comment'].apply(predict_sentiment)
                    st.session_state.summary = summarize_comments_langchain(st.session_state.comments, grooq_api_key)

    if page == "Home":
        st.title("\U0001F3A5 YouTube Sentiment Analyzer Dashboard")
        st.markdown("""
            Welcome to the **YouTube Sentiment Analyzer**! This tool helps you analyze comments from YouTube videos to understand viewer sentiments, generate insights, and interact with a CommentBot for detailed queries.
        """)

    elif page == "Analysis":
        st.title("\U0001F4C8 Analysis Results")
        if st.session_state.df is not None and 'Sentiment' in st.session_state.df.columns:
            visualize_sentiment(st.session_state.df)
            st.subheader("\U0001F4DD Summarized Insights")
            st.markdown(st.session_state.summary if st.session_state.summary else "No summary available.")
            st.subheader("\U0001F4CB Comment Data")
            st.download_button("\u2B07\ufe0f Download CSV", st.session_state.df.to_csv(index=False), "yt_sentiments.csv", "text/csv")
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        else:
            st.info("Please run the analysis from the sidebar to view results.")

    elif page == "CommentBot":
        st.title("\U0001F4AC CommentBot")
        if st.session_state.comments:
            st.session_state.user_query = st.text_input("Ask anything about the comments:", value=st.session_state.user_query)
            if st.button("Ask CommentBot", key="commentbot_button"):
                with st.spinner("\u23F3 Processing your question..."):
                    st.session_state.bot_response = qa_bot_response_langchain(
                        st.session_state.user_query, st.session_state.comments, grooq_api_key)
            if st.session_state.bot_response:
                st.markdown(f"**Bot Answer:** {st.session_state.bot_response}")
        else:
            st.info("Please run the analysis from the sidebar to enable CommentBot.")

    elif page == "About":
        st.title("\u2139\ufe0f About")
        st.markdown("""
        **YouTube Sentiment Analyzer** is a professional tool built with Streamlit, multilingual transformer models, and LangChain + GROQ.

        - **Sentiment Analysis**: Supports multilingual comment classification.
        - **Visual Insights**: Bar chart and word cloud visualization.
        - **AI-Powered Summary & Q&A**: Get high-level summaries and answers using LLMs.

        Built for creators, marketers, and analysts to gain audience insights.
        """)

    st.markdown("""
        <script> window.scrollTo(0, document.body.scrollHeight); </script>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
