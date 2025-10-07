from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from google import genai
import os
import logging

# Flask app setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.config.update(
    SESSION_COOKIE_SAMESITE='None',
    SESSION_COOKIE_SECURE=True,
)

# CORS setup
raw_frontend_origins = os.environ.get('FRONTEND_ORIGIN', '')
allowed_origins = [o.strip().lstrip('=') for o in raw_frontend_origins.split(',') if o.strip()]
if not allowed_origins:
    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

cors_config = {
    r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Set-Cookie"],
        "supports_credentials": True,
    }
}
CORS(app, resources=cors_config, supports_credentials=True)

# Logging
logger.info(f"Startup: PORT={os.environ.get('PORT')} binding via gunicorn")

# AI Configuration
_genai_client = None
model_name = "gemini-2.5-flash"

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client

# Global variables
conversation = []
user_subject = ""
first_message = None  # Variable to store the first message

# Helper functions
def User_append(user_input):
    conversation.append({"role": "user", "content": user_input})

def AI_append(ai_resp):
    conversation.append({"role": "Socrates", "content": ai_resp})

def get_answer(user_subject, user_input):
    prompt = f"""
    You are Socrates, a patient, insightful teacher guiding a curious student through deep reasoning. 
    Your only tool is questioning. You never give direct answers. You use thoughtful, targeted questions 
    to make the student analyze, compare, and evaluate ideas in their own words.

    Your goal: reveal the student’s reasoning process and identify any gaps or misconceptions. 
    Each question should:
    - Stay strictly within the scope of the user’s last message.
    - Encourage reflection on causes, consequences, and relationships.
    - Explore contrasts, edge cases, or hypothetical situations.
    - Help the user clarify or test their own understanding.

    If you sense confusion or incorrect reasoning, ask a question that gently exposes the flaw 
    without directly stating it. If the user struggles, simplify the question or provide a subtle hint.

    If the user demonstrates full understanding, return to the first message in {conversation} 
    and proceed to the next topic mentioned in {first_message}. 
    Only move forward if you are absolutely certain the user has mastered the current idea.

    You must output exactly one question per turn.
    Your Questions should follow the socratic method of questioning.

    Context:
    - Current topic: {user_subject}
    - User just said: {user_input}
    - Conversation so far: {conversation}
    - Original message: {first_message}
    """
    client = get_genai_client()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    ai_answer = response.text
    AI_append(ai_answer)
    return ai_answer

# Hardcoded users
USERS = {
    "Avnish": "Nerd",
    "Krish": "Newbie",
    "Swaroop": "Stupid",
    "Anuj": "SmartGuy",
    "Windstorm": "AlanMcBob",
    "Ms.Lerner": "Biology",
    "Ms.McCracken" : "Chemistry",
}

def handle_user_message(message):
    global first_message
    if first_message is None:
        first_message = message  # Record the first message
        print(f"First message recorded: {first_message}")
    else:
        print(f"User message: {message}")

def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Allow OPTIONS requests to pass through for CORS preflight
        if request.method == 'OPTIONS':
            return func(*args, **kwargs)
        if not session.get("authenticated"):
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    return wrapper

# Routes
@app.route('/')
def index():
    return render_template(
        'index.html',
        contact_text=os.environ.get('CONTACT_TEXT', 'Contact'),
        contact_url=os.environ.get('CONTACT_URL', '')
    )

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
        "working_dir": os.getcwd(),
        "frontend_origins": allowed_origins,
        "secret_key_present": bool(os.environ.get('SECRET_KEY'))
    }
    logger.info(f"/debug: {info}")
    return jsonify(info), 200

@app.route('/set_subject', methods=['POST', 'OPTIONS'])
@login_required
def set_subject():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        global user_subject
        global conversation
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        user_subject = data.get('subject', '')
        conversation = []
        logger.info(f"Subject set to: {user_subject}")
        return jsonify({"status": "success", "subject": user_subject})
    except Exception as e:
        logger.error(f"Error in set_subject: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/ask', methods=['POST', 'OPTIONS'])
@login_required
def ask():
    if request.method == 'OPTIONS':
        return '', 200
    global user_subject
    data = request.json
    user_input = data.get('message', '')
    handle_user_message(user_input)
    User_append(user_input)
    ai_answer = get_answer(user_subject, user_input)
    return jsonify({"answer": ai_answer})

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json(silent=True) or {}
    username = data.get('username', '')
    password = data.get('password', '')
    if username in USERS and USERS[username] == password:
        session['authenticated'] = True
        session['username'] = username
        return jsonify({"ok": True, "username": username})
    return jsonify({"ok": False, "error": "Invalid credentials"}), 401

@app.route('/logout', methods=['POST', 'OPTIONS'])
def logout():
    if request.method == 'OPTIONS':
        return '', 200
    session.clear()
    return jsonify({"ok": True})

@app.route('/me', methods=['GET'])
def me():
    if session.get('authenticated'):
        return jsonify({"authenticated": True, "username": session.get('username')})
    return jsonify({"authenticated": False}), 200

@app.route('/test', methods=['GET', 'POST', 'OPTIONS'])
def test():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({
        "status": "ok", 
        "method": request.method,
        "headers": dict(request.headers),
        "cors_working": True
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
