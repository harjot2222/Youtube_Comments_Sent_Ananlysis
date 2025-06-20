# 🎥 YouTube Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)
![LangChain](https://img.shields.io/badge/NLP-LangChain%20%2B%20GROQ-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A professional Streamlit app that analyzes YouTube video comments using **VADER Sentiment Analysis** and **LangChain + GROQ-powered LLMs**. Gain sentiment insights, generate summaries, and interact with an AI CommentBot to explore viewer feedback in detail.

---

## 🔍 Features

✅ Fetches comments from any YouTube video  
✅ Cleans and preprocesses text using NLP techniques  
✅ Classifies sentiment (Positive / Negative / Neutral) using VADER  
✅ Visualizes results using bar charts and word clouds  
✅ Generates AI-powered summaries using LangChain + GROQ  
✅ Interactive CommentBot for querying viewer feedback  
✅ Beautiful, responsive Streamlit UI with custom CSS

---

## 🚀 Demo

<p align="center">
  <img src="https://github.com/your-username/your-repo-name/assets/demo.gif" width="75%">
</p>

---

## 🧠 Tech Stack

- **Frontend**: Streamlit + HTML/CSS
- **Backend**: Python
- **APIs**: YouTube Data API v3, GROQ (via LangChain)
- **Libraries**: `pandas`, `matplotlib`, `seaborn`, `vaderSentiment`, `wordcloud`, `google-api-python-client`, `langchain`, `streamlit`

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/youtube-sentiment-analyzer.git
cd youtube-sentiment-analyzer
pip install -r requirements.txt

🔐 Setup: API Keys
Go to your Streamlit secrets file:
~/.streamlit/secrets.toml (or .streamlit/secrets.toml in the project root)

Add your keys like this:

toml
Copy
Edit
YOUTUBE_API_KEY = "your_youtube_api_key"
GROOQ_API_KEY = "your_groq_api_key"
Get YouTube API Key: https://console.developers.google.com

Get GROQ API Key: https://console.groq.com (requires xAI/GROQ account)

🧪 Run the App
bash
Copy
Edit
streamlit run app.py
🖼️ Screenshots
🔍 Sentiment Analysis


☁️ WordCloud


🤖 CommentBot


📦 Project Structure
bash
Copy
Edit
📁 youtube-sentiment-analyzer
│
├── app.py                     # Main Streamlit App
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── secrets.toml          # API Keys
├── assets/
│   └── youtube-logo.png      # Logo used in sidebar
└── README.md                 # You're reading this!
📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and share — with credit.

🙌 Credits
Built with ❤️ by Your Name
Using:

Streamlit

YouTube Data API

LangChain

GROQ + LLaMA 3

💡 Future Enhancements
🎯 Sentiment filtering (show only positive/negative)

📊 Time-series sentiment trends

🧵 Threaded comment analysis

🌍 Multi-language support

🤝 Contributions Welcome!
Pull requests, feature suggestions, or issues — all are welcome!
Let’s improve it together 💥

⭐ If you found this useful, give it a star!
It helps others discover the project 🌟

yaml
Copy
Edit

---

Let me know if you'd like me to:
- Add a `requirements.txt`
- Generate demo screenshots or a GIF preview
- Help you deploy to **Streamlit Cloud** or **Render**










