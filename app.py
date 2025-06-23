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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException

# Set Streamlit page configuration
st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")

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
        return lang
    except LangDetectException:
        return 'unknown'

def predict_sentiment_vader(comment):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(comment)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

@st.cache_data
def translate_to_english(comment, groq_api_key):
    try:
        model = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        prompt = PromptTemplate.from_template("""
            Translate the following comment to English. Provide only the translated text.

            Comment:
            {comment}

            Translated:
        """)
        chain = LLMChain(llm=model, prompt=prompt)
        response = chain.invoke({"comment": comment})
        translated_text = response['text'].strip()
        return translated_text if translated_text else comment
    except AuthenticationError:
        st.error("❌ Invalid Groq API key for translation.")
        return comment
    except Exception as e:
        st.warning(f"⚠️ Translation error: {str(e)}. Using original comment.")
        return comment

def predict_sentiment(comment, groq_api_key, use_multilingual):
    cleaned_comment = clean_text(comment)
    if not use_multilingual:
        return predict_sentiment_vader(cleaned_comment)
    else:
        lang = detect_language(cleaned_comment)
        if lang == 'en':
            return predict_sentiment_vader(cleaned_comment)
        else:
            translated_comment = translate_to_english(cleaned_comment, groq_api_key)
            return predict_sentiment_vader(translated_comment)

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
    all_comments = " ".join(results_df["Translated_Comment"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    st.subheader("📄 Sample Comments")
    st.download_button("⬇️ Download CSV", results_df.to_csv(index=False), "yt_comments.csv", "text/csv")
    st.dataframe(results_df[['Comment', 'Translated_Comment', 'Sentiment', 'Language']].head(10), use_container_width=True)

@st.cache_data
def summarize_comments_langchain(comments, groq_api_key):
    try:
        model = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        translated_comments = [translate_to_english(comment, groq_api_key) for comment in comments]
        prompt = PromptTemplate.from_template("""
            Summarize the following YouTube comments. Provide a short summary, highlight common themes, user concerns, and what users appreciated or disliked. Give actionable insights for improving future videos.

            Comments:
            {comments}
        """)
        chain = LLMChain(llm=model, prompt=prompt)
        response = chain.invoke({"comments": "\n".join(translated_comments[:100])})
        return response['text'].strip()
    except AuthenticationError:
        return "❌ Summary unavailable due to invalid Groq API key."
    except Exception as e:
        return f"⚠️ Error in summarization: {str(e)}."

def qa_bot_response_langchain(user_query, comments, groq_api_key):
    try:
        model = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        translated_comments = [translate_to_english(comment, groq_api_key) for comment in comments]
        prompt = PromptTemplate.from_template("""
            You are a helpful assistant analyzing YouTube comments. Use the following comments to answer the user's question:

            Comments:
            {comments}

            Question:
            {query}

            Answer:
        """)
        chain = LLMChain(llm=model, prompt=prompt)
        response = chain.invoke({"comments": "\n".join(translated_comments[:100]), "query": user_query})
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

    # Sidebar configuration
    with st.sidebar:
        st.image("youtube-logo-png-46020.png", use_container_width=True)
        st.header("🎬 YouTube Analyzer")
        page = st.selectbox("Navigate", ["Home", "Analysis", "CommentBot", "About"])

        if page in ["Home", "Analysis", "CommentBot"]:
            st.subheader("🔧 Analysis Settings")
            youtube_url = st.text_input("YouTube Video URL")
            max_comments = st.slider("Number of Comments", 50, 500, 200, step=50)
            multilingual_toggle = st.toggle("Enable Multilingual Sentiment Analysis (Grok for Translation)")

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
                    # Create DataFrame with original and translated comments
                    st.session_state.df = pd.DataFrame(st.session_state.comments, columns=["Comment"])
                    st.session_state.df['Language'] = st.session_state.df['Comment'].apply(detect_language)
                    st.session_state.df['Translated_Comment'] = st.session_state.df.apply(
                        lambda row: row['Comment'] if row['Language'] == 'en' else translate_to_english(row['Comment'], groq_api_key),
                        axis=1
                    )
                    st.session_state.df['Sentiment'] = st.session_state.df['Translated_Comment'].apply(
                        lambda c: predict_sentiment_vader(clean_text(c))
                    )
                    st.session_state.summary = summarize_comments_langchain(st.session_state.comments, groq_api_key)
                    st.success("✅ Analysis completed.")

    # Page rendering
    if page == "Home":
        st.title("🎯 YouTube Sentiment Analyzer")
        st.markdown("Analyze YouTube comments with VADER for English and Grok for translating non-English comments!")

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
            st.session_state.user_query = st.text_input("Ask something about the comments:", value=st.session_state.user_query)
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
        This app analyzes YouTube comments with:
        - **VADER** for English sentiment analysis
        - **Grok + LLaMA 3** for translating non-English comments to English before VADER analysis

        Built with ❤️ using Streamlit, LangChain, and open-source models.
        """)

    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
