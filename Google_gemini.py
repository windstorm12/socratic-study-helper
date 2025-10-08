from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from google import genai
import os
import logging
import json

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

# ---------------- Math mode helpers ----------------
def _wants_no_harder(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "no harder", "not harder", "same difficulty", "do not make harder",
        "don't make harder", "keep same level", "not give harder"
    ]
    return any(k in t for k in keywords)

def _extract_json_block(text: str) -> str:
    if not text:
        return ""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

def _get_decimal_places() -> int:
    try:
        return max(0, int(os.environ.get('MATH_DECIMALS', '2')))
    except Exception:
        return 2

def _normalize_numeric_string(value: float, places: int) -> str:
    s = f"{round(value, places):.{places}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

def _generate_math_set(concept: str, allow_harder: bool):
    if _wants_no_harder(concept):
        level_note = "keep them at the same difficulty as the previous set"
    else:
        level_note = "gradually increase difficulty from 1 (easiest) to 10 (hardest) and be slightly harder overall"
    prompt = f"""
    You are a math question generator.
    Create exactly 10 unique practice problems testing the SAME underlying concept as:
    "{concept}"

    Level note: {level_note}.

    Rules:
    - The 10 problems must gradually increase in difficulty from 1 (easiest) to 10 (hardest).
    - Include a mix of numeric, word-based, and symbolic problems when relevant.
    - If numeric, simplify answers. If symbolic, express answers cleanly (no long fractions or decimals).
    - Output STRICT JSON:
    {{
      "problems": [
        {{"number": 1, "question": "...", "answer": "..." }},
        ...
        {{"number": 10, "question": "...", "answer": "..." }}
      ]
    }}
    - No explanations, commentary, or markdown.
    """
    client = get_genai_client()
    response = client.models.generate_content(model=model_name, contents=prompt)
    raw = response.text
    try:
        json_text = _extract_json_block(raw)
        data = json.loads(json_text)
        problems = data.get("problems", [])
    except json.JSONDecodeError:
        logger.warning("Model returned bad JSON. Attempting recovery.")
        import re
        problems = []
        matches = re.findall(r"\d+\).*?answer.*?:.*?(?=\n\d+\)|$)", raw, re.S | re.I)
        for i, m in enumerate(matches[:10], start=1):
            q_match = re.search(r"\d+\)\s*(.*?)\s*answer", m, re.I)
            a_match = re.search(r"answer.*?:\s*(.*)", m, re.I)
            if q_match and a_match:
                problems.append({"number": i, "question": q_match.group(1).strip(), "answer": a_match.group(1).strip()})
    except Exception:
        problems = []
    formatted = []
    places = _get_decimal_places()
    for idx, p in enumerate(problems[:10], start=1):
        q = p.get("question") if isinstance(p, dict) else None
        a = p.get("answer") if isinstance(p, dict) else None
        if q and a is not None:
            try:
                num = float(str(a).replace(',', '').strip())
                a_str = _normalize_numeric_string(num, places)
            except Exception:
                a_str = str(a).strip()
            formatted.append({"number": idx, "question": q, "answer": a_str})
    return formatted

def _parse_user_answers(text: str) -> dict:
    import re
    answers = {}
    for part in re.split(r"[\n;,]", text or ""):
        m = re.search(r"(\d{1,2})\s*[\)\:\-]??\s*(.+)", part.strip())
        if m:
            num = int(m.group(1))
            ans = m.group(2).strip()
            if ans:
                answers[num] = ans
    if not answers:
        tokens = [t for t in (text or "").replace("\n", " ").split(" ") if t.strip()]
        for i, tok in enumerate(tokens[:10], start=1):
            answers[i] = tok
    return answers

def _answers_equal(expected: str, given: str) -> bool:
    import re
    def to_number_or_none(s: str):
        if s is None:
            return None
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except Exception:
                return None
        return None

    en = to_number_or_none(expected)
    gn = to_number_or_none(given)
    if en is not None and gn is not None:
        places = _get_decimal_places()
        return _normalize_numeric_string(en, places) == _normalize_numeric_string(gn, places)

    def clean(s: str) -> str:
        s = str(s).strip()
        return s

    return clean(expected) == clean(given)

def _generate_hint(question: str, correct_answer: str) -> str:
    try:
        client = get_genai_client()
        prompt = f"""
        The student answered a math problem incorrectly.
        Provide a short, encouraging hint (1–2 sentences) that guides them toward the solution
        without revealing the final answer. Keep it concise and actionable.

        Question: {question}
        Correct Answer: {correct_answer}
        """
        response = client.models.generate_content(model=model_name, contents=prompt)
        return (response.text or "Check your setup and isolate the variable step by step.").strip()
    except Exception:
        return "Re-check your steps and isolate the unknown carefully before substituting."

def _looks_like_answers(text: str) -> bool:
    import re
    if not text:
        return False
    numbered = re.findall(r"\b(\d{1,2})\s*[\)\:\-]", text)
    if len(numbered) >= 3:
        return True
    if sum(text.count(sep) for sep in [',', ';', '\n']) >= 3:
        return True
    return False

def _is_new_seed_request(text: str) -> bool:
    t = (text or "").lower()
    keywords = ["new set", "new problems", "another", "more", "different set", "next set", "again", "generate"]
    if any(k in t for k in keywords):
        return True
    has_math = ('=' in t) or ('?' in t)
    many_seps = sum(t.count(sep) for sep in [',', ';', '\n']) >= 3
    return has_math and not many_seps

def _wants_help(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["help", "explain", "steps", "how do i", "walk me through", "guide me"])

def _extract_problem_number(text: str) -> int:
    import re
    if not text:
        return -1
    m = re.search(r"problem\s*(\d{1,2})", text.lower())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    m2 = re.search(r"\b(\d{1,2})\b", text)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return -1
    return -1

def _generate_followups(concept: str):
    prompt = f"""
    Create exactly 5 focused practice problems on the SAME concept as "{concept}".
    Keep difficulty similar to the original missed problem. Output STRICT JSON:
    {{
      "problems": [
        {{"number": 1, "question": "...", "answer": "..."}},
        {{"number": 2, "question": "...", "answer": "..."}},
        {{"number": 3, "question": "...", "answer": "..."}},
        {{"number": 4, "question": "...", "answer": "..."}},
        {{"number": 5, "question": "...", "answer": "..."}}
      ]
    }}
    No commentary or markdown.
    """
    client = get_genai_client()
    response = client.models.generate_content(model=model_name, contents=prompt)
    raw = response.text
    try:
        data = json.loads(_extract_json_block(raw))
        probs = data.get("problems", [])
    except Exception:
        probs = []
    places = _get_decimal_places()
    formatted = []
    for idx, p in enumerate(probs[:5], start=1):
        q = p.get("question") if isinstance(p, dict) else None
        a = p.get("answer") if isinstance(p, dict) else None
        if q and a is not None:
            try:
                num = float(str(a).replace(',', '').strip())
                a_str = _normalize_numeric_string(num, places)
            except Exception:
                a_str = str(a).strip()
            formatted.append({"number": idx, "question": q, "answer": a_str})
    return formatted

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
        if user_subject == 'Math':
            session['math_mode'] = {
                'problems': [],
                'generated': False,
                'allow_harder': True
            }
        else:
            session.pop('math_mode', None)
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
    if session.get('math_mode') is not None and user_subject == 'Math':
        math_state = session.get('math_mode') or {}
        if not math_state.get('generated'):
            allow_harder = not _wants_no_harder(user_input)
            concept = (session.get('math_mode') or {}).get('concept') or user_input
            problems = _generate_math_set(concept, allow_harder)
            math_state['problems'] = problems
            math_state['generated'] = True
            math_state['allow_harder'] = allow_harder
            math_state['concept'] = concept
            session['math_mode'] = math_state
            if problems:
                lines = [f"{p['number']}) {p['question']}" for p in problems]
                response_text = (
                    "Here are 10 practice problems. When you're ready, reply with your answers in the form '1) answer, 2) answer, ...' or one per line.\n\n"
                    + "\n".join(lines)
                )
            else:
                response_text = "I couldn't generate problems right now. Please rephrase your question."
            AI_append(response_text)
            return jsonify({"answer": response_text})
        if user_input.strip().lower() == 'retry' and math_state.get('pending'):
            retry_lines = [f"{p['number']}) {p['question']}" for p in math_state['pending']]
            response_text = "Let's try these again. Remember: short, clear steps.\n\n" + "\n".join(retry_lines)
            AI_append(response_text)
            return jsonify({"answer": response_text})
        if _wants_help(user_input) and math_state.get('pending'):
            prob_num = _extract_problem_number(user_input)
            target = None
            if prob_num != -1:
                for p in math_state['pending']:
                    if p.get('number') == prob_num:
                        target = p
                        break
            target = target or (math_state['pending'][0] if math_state['pending'] else None)
            if target:
                hint = _generate_hint(target.get('question'), target.get('answer'))
                response_text = (
                    f"Let's walk through problem {target.get('number')}. {hint}\n\n"
                    "Try it again and send your updated answer."
                )
                AI_append(response_text)
                return jsonify({"answer": response_text})

        if _is_new_seed_request(user_input) and not _looks_like_answers(user_input):
            allow_harder = not _wants_no_harder(user_input)
            concept = (session.get('math_mode') or {}).get('concept') or user_input
            problems = _generate_math_set(concept, allow_harder)
            math_state['problems'] = problems
            math_state['generated'] = True
            math_state['allow_harder'] = allow_harder
            math_state['concept'] = concept
            session['math_mode'] = math_state
            if problems:
                lines = [f"{p['number']}) {p['question']}" for p in problems]
                response_text = (
                    "Here are 10 practice problems. When you're ready, reply with your answers in the form '1) answer, 2) answer, ...' or one per line.\n\n"
                    + "\n".join(lines)
                )
            else:
                response_text = "I couldn't generate problems right now. Please rephrase your question."
            AI_append(response_text)
            return jsonify({"answer": response_text})
        problems = math_state.get('problems', [])
        given_answers = _parse_user_answers(user_input)
        results = []
        correct_count = 0
        pending = []
        for p in problems:
            num = p.get('number')
            expected = p.get('answer')
            given = given_answers.get(num)
            if given is None:
                pending.append(p)
                results.append(f"{num}) Missing. We'll keep this for a retry.")
                continue
            ok = _answers_equal(expected, given)
            if ok:
                correct_count += 1
                results.append(f"{num}) Correct")
            else:
                hint = _generate_hint(p.get('question'), expected)
                results.append(f"{num}) Not quite right. Hint: {hint}")
                pending.append(p)
        summary = f"You got {correct_count}/{len(problems)} correct."
        if pending:
            session['math_mode']['pending'] = pending
            response_text = (
                summary
                + "\n"
                + "\n".join(results)
                + "\n\nYou have "
                + str(len(pending))
                + " question(s) to retry. Reply 'retry' to try them again, or ask for help on a specific problem (e.g., 'help with problem 4')."
            )
        else:
            session['math_mode']['pending'] = []
            response_text = summary + "\n" + "\n".join(results)
        AI_append(response_text)
        return jsonify({"answer": response_text})
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
