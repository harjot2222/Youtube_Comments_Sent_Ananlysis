🎥 YouTube Sentiment Analyzer
AI-powered Streamlit App using VADER, BERT, and LangChain + GROQ

Analyze and visualize YouTube video comments with sentiment classification, multilingual support, AI-generated summaries, and a smart CommentBot assistant.

🔍 Features
✅ Fetches comments from any public YouTube video
✅ Cleans and preprocesses text using NLP techniques
✅ Detects comment language (supports English, Hindi, and Hinglish)
✅ Performs sentiment classification using BERT (via Hugging Face Transformers)
✅ Visualizes sentiment distribution and word clouds
✅ Generates AI-powered summaries using LangChain + GROQ (LLaMA 3)
✅ Chat with CommentBot to query viewer feedback using LLMs
✅ Fully responsive UI with modern Streamlit styling and custom CSS

🚀 Live Demo
👉 (Add link here once deployed — e.g., Streamlit Cloud, Render)

🧠 Tech Stack
Layer	Technologies
Frontend	Streamlit + HTML/CSS
Backend	Python
APIs	YouTube Data API v3, GROQ (via LangChain)
Libraries	pandas, matplotlib, seaborn, wordcloud, vaderSentiment, langchain, streamlit, google-api-python-client, transformers, langdetect

⚙️ Installation
bash
Copy
Edit
git clone https://github.com/your-username/youtube-sentiment-analyzer.git
cd youtube-sentiment-analyzer
pip install -r requirements.txt
🔐 Setup: API Keys
Add your keys to secrets.toml (create this if it doesn't exist):

Location:

Linux/Mac: ~/.streamlit/secrets.toml

Windows/Project: .streamlit/secrets.toml

toml
Copy
Edit
YOUTUBE_API_KEY = "your_youtube_api_key"
GROQ_API_KEY = "your_groq_api_key"
🔑 Get YouTube API Key

🔑 Get GROQ API Key (xAI/GROQ account required)

🧪 Run the App
bash
Copy
Edit
streamlit run app.py
📸 Screenshots
Sentiment Analysis	Word Cloud	CommentBot

💡 Add these screenshots in a screenshots/ folder.

📂 Project Structure
bash
Copy
Edit
📁 youtube-sentiment-analyzer/
│
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── secrets.toml          # API keys configuration
├── assets/
│   └── youtube-logo.png      # Sidebar logo
├── screenshots/              # Demo images (optional)
└── README.md                 # This file
📄 License
Licensed under the MIT License.
Feel free to use, modify, or share — just give proper credit.

🙌 Credits
Built with ❤️ by [Your Name]
Powered by:

Streamlit

YouTube Data API

LangChain

GROQ + LLaMA 3

Hugging Face Transformers

💡 Future Enhancements
🎯 Sentiment filter (Positive / Negative toggle)

📊 Time-series trend analysis

🧵 Threaded comment analysis

🌍 Multi-language translation support

🧠 Topic modeling for clustering comment themes

🤝 Contributions Welcome!
Pull requests, issues, and ideas are welcome!
Let’s build something insightful and impactful together 💥

⭐ If you find this project helpful, give it a star!
It helps others discover and use it 🌟

