from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from google import genai
import os
import logging

# ----------------------
# Flask app
# ----------------------
app = Flask(__name__)
# Secret key for session cookies
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
# Session cookies for cross-site (Vercel → Railway)
app.config.update(
    SESSION_COOKIE_SAMESITE='None',
    SESSION_COOKIE_SECURE=True,
)
# Enable CORS with credentials and restrict to frontend origin
frontend_origin = os.environ.get('FRONTEND_ORIGIN')  # e.g. https://your-app.vercel.app
if frontend_origin:
    CORS(app, supports_credentials=True, resources={r"/*": {"origins": [frontend_origin]}})
else:
    # Fallback allows localhost dev; update FRONTEND_ORIGIN in production
    CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log bound PORT on startup
logger.info(f"Startup: PORT={os.environ.get('PORT')} binding via gunicorn")
try:
    # Log registered routes for debugging
    from werkzeug.routing import Rule
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    logger.info(f"Registered routes: {routes}")
except Exception as e:
    logger.warning(f"Could not list routes: {e}")

@app.before_request
def log_request_info():
    logger.info(f"Incoming request: {request.method} {request.path}")

# ----------------------
# Configure AI (lazy init)
# ----------------------
_genai_client = None
model_name = "gemini-1.5-flash"  # update if needed

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            # Delay failure until first AI usage; health and static routes should work
            raise ValueError("GEMINI_API_KEY environment variable is required")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client

# ----------------------
# Global variables
# ----------------------
conversation = []
user_subject = ""

# ----------------------
# Helper functions
# ----------------------
def User_append(user_input):
    conversation.append({"role": "user", "content": user_input})

def AI_append(ai_resp):
    conversation.append({"role": "Socrates", "content": ai_resp})

def get_answer(user_subject, user_input):
    prompt = f"""
    You are Socrates, a master teacher guiding a student through deep critical thinking. 
    Your goal is to make the student reason carefully, analyze, and reflect, not just recall facts. Each of your question should help identify
    the learning gaps the user has. If you see any incorrect explanation your question should indirectly convey that the user might be incorrect.
    If you notice any incorrect explanations, ask the user some critical thinking questions that challenge their understanding. If the user 
    is not able to answer the question, then help the user with a bit easier question and a hint.
    Ask questions that: 
    - Explore causes, consequences, and relationships. 
    - Compare and contrast ideas. 
    - Consider hypothetical scenarios or exceptions. 
    - Encourage evaluation and reasoning. 
    Stay strictly within the scope of the teaching material provided. 
    Do not give direct answers. Be curious, patient, and encouraging.
    You can only generate 1 question
    The user is talking about {user_subject} and they just said {user_input}, the conversation history is {conversation}
    """
    # Generate AI response using the client
    client = get_genai_client()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    ai_answer = response.text
    AI_append(ai_answer)
    return ai_answer

# ----------------------
# Auth utilities and Routes
# ----------------------

# Hardcoded users
USERS = {
    "Avnish": "nerd",
    "Krish": "gangster",
    "Ashwanth": "black",
    "swaroop": "annoying",
}

def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    return wrapper

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    logger.info("/health called")
    return jsonify({"status": "ok"}), 200

@app.route('/healthz', methods=['GET'])
def healthz():
    logger.info("/healthz called")
    return "ok", 200

@app.route('/debug', methods=['GET'])
def debug():
    info = {
        "port": os.environ.get('PORT'),
        "gemini_api_key_present": bool(os.environ.get('GEMINI_API_KEY')),
        "working_dir": os.getcwd()
    }
    logger.info(f"/debug: {info}")
    return jsonify(info), 200

@app.route('/set_subject', methods=['POST'])
@login_required
def set_subject():
    global user_subject
    data = request.json
    user_subject = data.get('subject', '')
    return jsonify({"status": "success", "subject": user_subject})

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    global user_subject
    data = request.json
    user_input = data.get('message', '')
    User_append(user_input)
    ai_answer = get_answer(user_subject, user_input)
    return jsonify({"answer": ai_answer})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get('username', '')
    password = data.get('password', '')
    # Validate
    if username in USERS and USERS[username] == password:
        session['authenticated'] = True
        session['username'] = username
        return jsonify({"ok": True, "username": username})
    return jsonify({"ok": False, "error": "Invalid credentials"}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"ok": True})

@app.route('/me', methods=['GET'])
def me():
    if session.get('authenticated'):
        return jsonify({"authenticated": True, "username": session.get('username')})
    return jsonify({"authenticated": False}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
