# Simple Flask app for Railway deployment
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
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
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Set-Cookie"],
        "supports_credentials": True,
    }
}
CORS(app, resources=cors_config, supports_credentials=True)

# Logging
logger.info(f"Startup: PORT={os.environ.get('PORT')} binding via gunicorn")

# Hardcoded users
USERS = {
    "Avnish": "Nerd",
    "Krish": "Newbie",
    "Ashwanth": "Black Monkey",
    "swaroop": "Stupid",
}

def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    return wrapper

# Routes
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
        "working_dir": os.getcwd(),
        "frontend_origins": allowed_origins,
        "secret_key_present": bool(os.environ.get('SECRET_KEY'))
    }
    logger.info(f"/debug: {info}")
    return jsonify(info), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get('username', '')
    password = data.get('password', '')
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

# Expose as "app" for gunicorn's wsgi entrypoint

