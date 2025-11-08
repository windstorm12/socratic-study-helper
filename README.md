# AI Socratic Tutor

An AI-powered tutoring system that helps students learn through guided questioning, with teacher monitoring capabilities.

## 🎯 What it does
- Explains concepts clearly using Groq's Llama 3.3
- Asks verification questions to ensure understanding (max 2 per topic)
- Teacher dashboard with student progress tracking
- Struggle detection and automated alerts
- Free to run (uses free-tier Groq API)

## 🛠️ Tech Stack
- **Backend:** Flask (Python)
- **AI:** Groq API (Llama 3.3 70B)
- **Database:** Supabase
- **Auth:** Supabase Auth
- **Frontend:** Vanilla HTML/CSS/JS with KaTeX for math rendering

## 📦 Setup

1. Clone the repo
```bash
git clone https://github.com/winstorm12/socratic-study-helper
cd socratic-ai-tutor

Install dependencies
pip install -r requirements.txt

Set environment variables
export GROQ_API_KEY='your_key_here'
export SUPABASE_URL='your_url'
export SUPABASE_ANON_KEY='your_key'
export SECRET_KEY='random_secret'

Run
python app.py

🎓 What I learned
Groq/Llama 3.3 follows instructions better than Gemini for educational use
Verification questions need strict limits (2 max) to avoid student frustration
Ed-tech paradox: students want ease, teachers want verification
Teacher tools ≠ student tools (can't optimize for both)
🚧 Known limitations
Currently works best for STEM subjects
No mobile app
Limited gamification
Teacher adoption requires direct outreach
🤝 Contributing
This is now in maintenance mode, but PRs welcome for:

Bug fixes
UI improvements
Additional subject support
Better mobile experience
📝 License
MIT License - do whatever you want with it

💭 Why I built this
Started as an attempt to build "better than ChatGPT" for learning, realized that's the wrong goal. Teachers need verification, students want speed. This sits in the middle - useful for classroom contexts where monitoring matters.