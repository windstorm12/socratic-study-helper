from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
import os
import logging

# ----------------------
# Flask app
# ----------------------
app = Flask(__name__)
CORS(app)  # <-- enables CORS for all routes

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Routes
# ----------------------

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

@app.route('/set_subject', methods=['POST'])
def set_subject():
    global user_subject
    data = request.json
    user_subject = data.get('subject', '')
    return jsonify({"status": "success", "subject": user_subject})

@app.route('/ask', methods=['POST'])
def ask():
    global user_subject
    data = request.json
    user_input = data.get('message', '')
    User_append(user_input)
    ai_answer = get_answer(user_subject, user_input)
    return jsonify({"answer": ai_answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
