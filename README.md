<h1 align="center">🎥 YouTube Sentiment Analyzer</h1> <p align="center"> 🔍 <strong>Analyze YouTube video comments with AI-powered Sentiment Analysis, Visualizations, Summarization, and an Interactive CommentBot!</strong> </p> <p align="center"> <img src="https://img.shields.io/badge/Made%20With-Streamlit-%23ff4b4b" /> <img src="https://img.shields.io/badge/NLP-BERT-blue" /> <img src="https://img.shields.io/badge/LLM-LangChain%20+%20GROQ-yellow" /> <img src="https://img.shields.io/badge/License-MIT-green" /> </p>
<h2>URL-> (https://sentitoolyt.streamlit.app/)</h2>
✨ Overview
This professional-grade Streamlit web app fetches and analyzes comments from any YouTube video using BERT-based sentiment analysis, visual tools like bar charts and word clouds, and advanced LangChain + GROQ-powered LLMs to summarize and interact with viewer feedback.

🔥 Key Features
📺 Fetch Comments from any public YouTube video

🧹 Clean & Preprocess Text using robust NLP techniques

😄 Classify Sentiment (Positive / Neutral / Negative) using BERT

📊 Visualize Results via Bar Charts and Word Clouds

🧠 Summarize Feedback using LangChain + GROQ (LLaMA 3)

🤖 Chat with AI CommentBot to explore viewer concerns, questions & suggestions

🎨 Responsive & Styled UI with beautiful custom CSS themes

🚀 Live Demo
🧪 Coming Soon!

Want help deploying to Streamlit Cloud? Just ask!

🧠 Tech Stack
Layer	Technology
Frontend	Streamlit, HTML/CSS
Backend	Python, LangChain
APIs Used	YouTube Data API v3, GROQ API
Libraries	pandas, matplotlib, seaborn, wordcloud, langdetect, transformers, google-api-python-client, vaderSentiment

⚙️ Installation
bash
Copy
Edit
git clone https://github.com/your-username/youtube-sentiment-analyzer.git
cd youtube-sentiment-analyzer
pip install -r requirements.txt
🔐 API Key Setup
Create a .streamlit/secrets.toml file and insert:

toml
Copy
Edit
YOUTUBE_API_KEY = "your_youtube_api_key"
GROQ_API_KEY = "your_groq_api_key"
🔑 Get YouTube API Key
🔑 Get GROQ API Key

🧪 How to Run
bash
Copy
Edit
streamlit run app.py
Then open http://localhost:8501 in your browser.

📸 Screenshots
Sentiment Summary	Word Cloud	CommentBot

📷 Add screenshots in screenshots/ folder

🗂️ Project Structure
bash
Copy
Edit
📁 youtube-sentiment-analyzer
├── app.py                    # Main Streamlit App
├── requirements.txt          # Python Dependencies
├── .streamlit/
│   └── secrets.toml          # API Keys (local only)
├── assets/
│   └── youtube-logo.png      # UI Branding
├── screenshots/              # App Demo Images
└── README.md                 # This file
📄 License
This project is licensed under the MIT License.
Feel free to use, distribute, and modify — just provide attribution ⭐

🙏 Credits
Made with ❤️ by Harjot Singh
Powered by:

Streamlit

LangChain

GROQ

HuggingFace Transformers

YouTube Data API

🌟 Future Enhancements
🎯 Sentiment filtering (Positive/Negative/Neutral only)

📊 Time-based sentiment trends

🧵 Threaded comment analysis

🌍 Multi-language comment support

🗃️ Comment clustering & topic modeling

🤝 Contribute
Got ideas or improvements? PRs and issues are welcome!

bash
Copy
Edit
# Fork and clone repo
# Make your changes
# Submit a Pull Request 🚀
If you love this project, don’t forget to ⭐ it and share with friends!

