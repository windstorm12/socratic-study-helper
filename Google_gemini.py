from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from google import genai
import os
import logging
import base64
import hashlib
from cryptography.fernet import Fernet, InvalidToken
import psycopg
import csv
from io import StringIO

# ----------------------
# Flask app
# ----------------------
"""
Ensure logging is initialized before any usage below (e.g., during CORS setup).
"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Secret key for session cookies
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
# Session cookies for cross-site (Vercel → Railway)
app.config.update(
    SESSION_COOKIE_SAMESITE='None',
    SESSION_COOKIE_SECURE=True,
)
# Enable CORS with credentials and restrict to frontend origin(s)
# Support multiple origins via comma-separated FRONTEND_ORIGIN env
raw_frontend_origins = os.environ.get('FRONTEND_ORIGIN', '')  # e.g. "https://app.vercel.app,https://preview.vercel.app"
allowed_origins = [o.strip().lstrip('=') for o in raw_frontend_origins.split(',') if o.strip()]
if not allowed_origins:
    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

logger.info(f"CORS allowed origins: {allowed_origins}")

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

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logger

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
# Optional: Initialize database schema on startup
# ----------------------
def _db_init_schema_if_needed():
    dsn = os.environ.get('DATABASE_URL')
    if not dsn:
        return
    conn = None
    try:
        conn = psycopg.connect(dsn, connect_timeout=5)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                      id SERIAL PRIMARY KEY,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                      username TEXT,
                      subject TEXT,
                      session_id TEXT,
                      sender TEXT CHECK (sender IN ('user','ai')) NOT NULL,
                      text TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
                    """
                )
        logger.info("DB schema ensured (messages table)")
    except Exception as e:
        logger.warning(f"DB schema init skipped (reason: {e})")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

# Ensure schema once at import time
_db_init_schema_if_needed()

# ----------------------
# Configure AI (lazy init)
# ----------------------
_genai_client = None
model_name = "gemini-2.5-flash"  # updated to available model name

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
    _db_log_message(sender="user", text=user_input)

def AI_append(ai_resp):
    conversation.append({"role": "Socrates", "content": ai_resp})
    _db_log_message(sender="ai", text=ai_resp)

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
# Database helpers
# ----------------------

def _get_db_conn():
    dsn = os.environ.get('DATABASE_URL')
    if not dsn:
        return None
    try:
        return psycopg.connect(dsn, connect_timeout=5)
    except Exception as e:
        logger.warning(f"DB connect failed: {e}")
        return None

def _db_log_message(sender: str, text: str):
    conn = _get_db_conn()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (username, subject, session_id, sender, text)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        session.get('username'),
                        user_subject or None,
                        request.cookies.get('session'),
                        sender,
                        text,
                    ),
                )
    except Exception as e:
        logger.warning(f"DB insert failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _db_export_all_as_csv():
    conn = _get_db_conn()
    if conn is None:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at, username, subject, session_id, sender, text
                    FROM messages
                    ORDER BY created_at ASC
                    """
                )
                rows = cur.fetchall()
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["created_at", "username", "subject", "session_id", "sender", "text"])
        for r in rows:
            writer.writerow(r)
        return output.getvalue()
    except Exception as e:
        logger.warning(f"DB export failed: {e}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ----------------------
# Auth utilities and Routes
# ----------------------

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
        "database_url_present": bool(os.environ.get('DATABASE_URL')),
        "secret_key_present": bool(os.environ.get('SECRET_KEY'))
    }
    logger.info(f"/debug: {info}")
    return jsonify(info), 200

@app.route('/set_subject', methods=['POST'])
@login_required
def set_subject():
    global user_subject
    global conversation
    data = request.json
    user_subject = data.get('subject', '')
    # Reset server-side conversation when subject changes
    conversation = []
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

# ----------------------
# Encryption helpers and endpoints
# ----------------------

def _get_fernet() -> Fernet:
    secret = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
    # Derive a 32-byte urlsafe key from SECRET_KEY deterministically
    digest = hashlib.sha256(secret.encode('utf-8')).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)

@app.route('/encrypt', methods=['POST'])
@login_required
def encrypt_payload():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Missing JSON body"}), 400
    f = _get_fernet()
    token = f.encrypt(jsonify(data).get_data())
    # Return as base64 text
    return jsonify({"ciphertext": token.decode('utf-8')}), 200

@app.route('/decrypt', methods=['POST'])
@login_required
def decrypt_payload():
    payload = request.get_json(silent=True) or {}
    ciphertext = payload.get('ciphertext', '')
    if not ciphertext:
        return jsonify({"error": "Missing ciphertext"}), 400
    f = _get_fernet()
    try:
        plaintext = f.decrypt(ciphertext.encode('utf-8'))
        # plaintext is raw JSON bytes that we produced via jsonify; return as passthrough JSON
        from flask import Response
        return Response(response=plaintext, status=200, mimetype='application/json')
    except InvalidToken:
        return jsonify({"error": "Invalid or corrupted ciphertext"}), 400

@app.route('/admin/export', methods=['GET'])
def admin_export():
    token = request.args.get('token')
    expected = os.environ.get('ADMIN_TOKEN')
    if not expected or token != expected:
        return jsonify({"error": "Forbidden"}), 403
    fmt = (request.args.get('format') or 'csv').lower()
    if fmt != 'csv':
        return jsonify({"error": "Only csv supported"}), 400
    data = _db_export_all_as_csv()
    if data is None:
        return jsonify({"error": "Export unavailable"}), 503
    from flask import Response
    return Response(
        data,
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename="conversations.csv"'
        }
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
