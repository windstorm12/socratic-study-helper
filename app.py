# app.py
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import google.generativeai as genai
import os, logging, json, random, string, datetime as dt
from collections import Counter
from datetime import datetime, timedelta
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-me')

# CORS setup
raw_frontend_origins = os.environ.get('FRONTEND_ORIGIN', '')
allowed_origins = [o.strip().lstrip('=') for o in raw_frontend_origins.split(',') if o.strip()]
if not allowed_origins:
    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

cors_config = {
    r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
    }
}
CORS(app, resources=cors_config, supports_credentials=True)

FORMATTING_GUIDE = """

FORMATTING:
- Use **bold** for key terms
- Use *italic* for emphasis
- Use LaTeX for all math:
  - Inline math: $F = ma$, $E = mc^2$
  - Display math (centered): $$F_c = \\frac{mv^2}{r}$$
  
LATEX EXAMPLES:
- Fractions: $\\frac{2x+5}{x-3}$ or $$\\frac{A}{x-3} + \\frac{B}{x+4}$$
- Exponents: $x^2$, $e^{-x}$
- Subscripts: $F_N$, $v_0$
- Square roots: $\\sqrt{x}$, $\\sqrt{x^2 + y^2}$
- Greek letters: $\\alpha$, $\\beta$, $\\pi$, $\\Delta$

EXAMPLE RESPONSE:
"To decompose $\\frac{2x+5}{(x-3)(x+4)}$, set it up as:

$$\\frac{A}{x-3} + \\frac{B}{x+4}$$

Where $A$ and $B$ are constants you'll solve for."
"""

def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if request.method == 'OPTIONS':
            return func(*args, **kwargs)
        if not session.get("authenticated"):
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    return wrapper

def teacher_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if request.method == 'OPTIONS':
            return func(*args, **kwargs)
        authorized = session.get("authenticated") and session.get("role") == "teacher"
        if not authorized:
            wants_json = 'application/json' in (request.headers.get('Accept','') or '')
            if wants_json or request.path.startswith('/api/'):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("teacher_login_page"))
        return func(*args, **kwargs)
    return wrapper

# AI Configuration
_genai_client = None
model_name = "gemini-2.5-flash"

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY required")
        genai.configure(api_key=api_key)
        _genai_client = True  # Just a flag that it's configured
    return genai

# Session-based conversation storage
def get_conversation():
    if 'conversation' not in session:
        session['conversation'] = []
    return session['conversation']

def get_subject():
    return session.get('subject', '')

def set_subject(subject):
    session['subject'] = subject
    session['conversation'] = []

# AI helpers
# app.py - Replace the get_answer function and add these new ones

def get_answer(user_subject, user_input):
    """Main function to generate AI teaching response"""
    
    # Get conversation history and settings
    conversation = get_conversation()
    teaching_mode = session.get('teaching_mode', 'guided')
    student_id = session.get('user_id')
    classroom_id = session.get('classroom_id')
    
    # Detect if this looks like a homework question
    homework_indicators = [
        'solve', 'calculate', 'what is the answer', 'help me with this problem',
        'do this', 'find the', 'determine', 'compute', 'what\'s the answer to',
        'answer to question', 'homework', 'assignment', 'due tomorrow',
        'calculate the value', 'solve for', 'find x', 'what is x'
    ]
    
    is_likely_homework = any(indicator in user_input.lower() for indicator in homework_indicators)
    has_numbers = any(char.isdigit() for char in user_input)
    
    # Get base teaching prompt for the mode
    base_prompt = get_teaching_prompt(teaching_mode)
    
    # Add homework warning if detected
    if is_likely_homework and has_numbers:
        homework_warning = """

⚠️ THIS LOOKS LIKE A HOMEWORK PROBLEM - DO NOT SOLVE IT!
Instead:
1. Ask what they've tried
2. Teach the method with a DIFFERENT example
3. Guide them to solve it themselves
4. Never give the numerical answer to their specific problem"""
        base_prompt += homework_warning
    
    # Add personalization if student is logged in
    if student_id:
        base_prompt = build_personalized_prompt(student_id, base_prompt)
        
        # Check for struggle and adapt approach
        if subject and classroom_id:
            is_struggling = detect_struggle(student_id, classroom_id, user_subject, conversation)
            
            if is_struggling:
                base_prompt += """

🔄 STRUGGLE DETECTED: Student has asked about this topic multiple times.
CHANGE YOUR STRATEGY:
1. Step back - check if they understand the prerequisites
2. Try a completely different explanation approach (visual, analogy, step-by-step)
3. Use Socratic questions to find the SPECIFIC gap in understanding
4. Consider they may have a fundamental misconception blocking progress
DO NOT repeat the same type of explanation you've been using."""
    
    # Build full prompt with conversation context
    conversation_context = conversation[-6:] if len(conversation) > 6 else conversation
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_context])
    
    full_prompt = f"""{base_prompt}

CURRENT CONTEXT:
Subject: {user_subject}
Student's question: "{user_input}"

Recent conversation:
{conversation_text}

Your response (remember to use LaTeX for all math):"""
    
    # Get AI response
    genai_client = get_genai_client()
    model = genai_client.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    ai_answer = response.text
    
    # Update conversation history
    conversation.append({"role": "user", "content": user_input})
    conversation.append({"role": "ai", "content": ai_answer})
    session['conversation'] = conversation
    
    # Update student profile periodically (every 3 exchanges)
    if student_id and len(conversation) % 6 == 0:
        try:
            update_student_profile_from_conversation(student_id, conversation)
            logger.info(f"Updated profile for student {student_id}")
        except Exception as e:
            logger.error(f"Error updating student profile: {e}")
    
    return ai_answer


def get_teaching_prompt(teaching_mode):
    """Get base teaching prompt based on mode (includes FORMATTING_GUIDE)"""
    
    if teaching_mode == 'socratic':
        base = """You are Socrates, a master teacher who guides through questions.
        
CRITICAL RULES:
- NEVER solve homework problems directly
- NEVER give direct answers to calculation questions
- If asked to solve/calculate, teach the METHOD instead
- Ask what they've tried first
- Guide them step-by-step to solve it themselves

Ask ONE thought-provoking question that helps the student discover the answer themselves."""
        
    elif teaching_mode == 'direct':
        base = """You are a helpful tutor who explains concepts clearly.

CRITICAL RULES:
- NEVER solve homework problems directly
- NEVER give direct numerical answers to their specific problems
- If they ask you to solve something, say: "I can't solve it for you, but I can teach you how!"
- Explain the METHOD and PROCESS clearly
- Give a DIFFERENT example (not their exact problem)
- Then ask them to try their problem using the method you taught

Teach concepts and methods, not answers."""
        
    else:  # guided (default)
        base = """You are an adaptive tutor who teaches effectively without doing the work for students.

CRITICAL RULES - NEVER BREAK THESE:
1. NEVER solve their homework problems directly
2. NEVER give direct answers to "solve this" or "calculate this" questions
3. If they ask you to solve/calculate, respond with: "I can't solve it for you, but I can teach you how to solve it yourself!"

YOUR TEACHING APPROACH:
1. If it's a homework-like question (solve, calculate, find):
   - First ask: "What have you tried so far?"
   - Teach the general method/concept
   - Give a DIFFERENT example (not their problem)
   - Guide them to solve THEIR problem step-by-step
   
2. If it's a concept question (what is, how does, why):
   - Give a clear explanation with examples
   - Check their understanding with a follow-up question
   
3. If they're stuck:
   - Ask what specific part is confusing
   - Break it into smaller steps
   - Guide, don't solve

Be encouraging but firm: you teach methods, students solve problems."""
    
    # Append the formatting guide to all modes
    return base + "\n" + FORMATTING_GUIDE
# ============ MEMORY & PERSONALIZATION ============

def get_student_profile(student_id, classroom_id):
    """Get or create student learning profile from Supabase"""
    from supabase import create_client
    
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_ANON_KEY')
    supabase = create_client(supabase_url, supabase_key)
    
    # Try to get existing profile
    response = supabase.table('student_profiles').select('*').eq('student_id', student_id).execute()
    
    if response.data and len(response.data) > 0:
        return response.data[0]
    else:
        # Create new profile
        new_profile = {
            'student_id': student_id,
            'strengths': [],
            'struggles': [],
            'mastered_concepts': [],
            'current_topics': {},
            'misconceptions': [],
            'learning_pace': 'average',
            'preferred_style': 'mixed',
            'total_sessions': 0,
            'avg_questions_to_understand': 5
        }
        result = supabase.table('student_profiles').insert(new_profile).execute()
        return result.data[0] if result.data else new_profile


def update_student_profile_from_conversation(student_id, conversation_history):
    """Analyze recent conversation and update student profile"""
    from supabase import create_client
    
    if len(conversation_history) < 4:
        return  # Need enough conversation to analyze
    
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_ANON_KEY')
    supabase = create_client(supabase_url, supabase_key)
    
    # Build analysis prompt
    recent_conv = conversation_history[-6:]  # Last 3 exchanges
    conv_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_conv])
    
    analysis_prompt = f"""Analyze this tutoring conversation and extract learning insights.

Conversation:
{conv_text}

Provide a JSON response with:
{{
  "understood_concepts": ["concept1", "concept2"],
  "struggled_with": ["concept3"],
  "misconceptions": ["misconception if any"],
  "learning_style_preference": "visual|verbal|step-by-step|analogies"
}}

Only include what's clearly evident. Return valid JSON only."""

    try:
        genai_client = get_genai_client()
        model = genai_client.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        import json
        analysis_text = response.text
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if json_match:
            insights = json.loads(json_match.group())
            
            # Get current profile
            profile = get_student_profile(student_id, None)
            
            # Update profile
            updates = {}
            
            if 'understood_concepts' in insights and insights['understood_concepts']:
                current_mastered = set(profile.get('mastered_concepts', []))
                current_mastered.update(insights['understood_concepts'])
                updates['mastered_concepts'] = list(current_mastered)
            
            if 'struggled_with' in insights and insights['struggled_with']:
                current_struggles = set(profile.get('struggles', []))
                current_struggles.update(insights['struggled_with'])
                updates['struggles'] = list(current_struggles)
            
            if 'misconceptions' in insights and insights['misconceptions']:
                current_misconceptions = set(profile.get('misconceptions', []))
                current_misconceptions.update(insights['misconceptions'])
                updates['misconceptions'] = list(current_misconceptions)
            
            if 'learning_style_preference' in insights:
                updates['preferred_style'] = insights['learning_style_preference']
            
            # Increment session count
            updates['total_sessions'] = profile.get('total_sessions', 0) + 1
            updates['updated_at'] = datetime.utcnow().isoformat()
            
            # Update in Supabase
            supabase.table('student_profiles').update(updates).eq('student_id', student_id).execute()
            
    except Exception as e:
        logger.error(f"Error updating student profile: {e}")


# ============ STRUGGLE DETECTION ============

def detect_struggle(student_id, classroom_id, current_topic, conversation_history):
    """Detect if student is struggling and create alerts"""
    from supabase import create_client
    
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_ANON_KEY')
    supabase = create_client(supabase_url, supabase_key)
    
    # Count how many times this topic appeared in recent history
    recent_messages = conversation_history[-10:]  # Last 10 messages
    topic_count = sum(1 for msg in recent_messages if msg.get('role') == 'user' and current_topic.lower() in msg.get('content', '').lower())
    
    # Check for frustration keywords
    frustration_keywords = ['don\'t get it', 'don\'t understand', 'confused', 'doesn\'t make sense', 'too hard', 'stuck']
    recent_user_messages = [msg.get('content', '').lower() for msg in recent_messages if msg.get('role') == 'user']
    frustration_detected = any(keyword in msg for msg in recent_user_messages for keyword in frustration_keywords)
    
    # Calculate time spent (rough estimate based on message count)
    time_estimate = len(recent_messages) * 2  # Assume ~2 min per exchange
    
    # Check if struggle event already exists
    one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
    existing = supabase.table('struggle_events').select('*').eq('student_id', student_id).eq('topic', current_topic).gte('created_at', one_hour_ago).execute()
    
    if topic_count >= 3 or frustration_detected:
        # Student is struggling
        
        if existing.data and len(existing.data) > 0:
            # Update existing event
            event_id = existing.data[0]['id']
            supabase.table('struggle_events').update({
                'attempt_count': topic_count,
                'frustration_detected': frustration_detected,
                'time_spent_minutes': time_estimate
            }).eq('id', event_id).execute()
        else:
            # Create new struggle event
            supabase.table('struggle_events').insert({
                'student_id': student_id,
                'classroom_id': classroom_id,
                'topic': current_topic,
                'attempt_count': topic_count,
                'time_spent_minutes': time_estimate,
                'frustration_detected': frustration_detected
            }).execute()
        
        # Create teacher alert if severe
        if topic_count >= 4 or frustration_detected:
            # Check if alert already exists
            alert_exists = supabase.table('teacher_alerts').select('*').eq('student_id', student_id).eq('topic', current_topic).eq('acknowledged', False).execute()
            
            if not alert_exists.data or len(alert_exists.data) == 0:
                # Get student name
                student = supabase.table('students').select('username').eq('id', student_id).execute()
                student_name = student.data[0]['username'] if student.data else 'Student'
                
                severity = 'high' if frustration_detected else 'medium'
                message = f"{student_name} is struggling with {current_topic}"
                recommendation = f"Student has attempted {topic_count} times. Consider reviewing {current_topic} in class or providing 1-on-1 help."
                
                supabase.table('teacher_alerts').insert({
                    'classroom_id': classroom_id,
                    'student_id': student_id,
                    'alert_type': 'struggle',
                    'topic': current_topic,
                    'severity': severity,
                    'message': message,
                    'recommendation': recommendation
                }).execute()
                
                logger.info(f"Created teacher alert for {student_name} struggling with {current_topic}")
        
        return True  # Is struggling
    
    return False  # Not struggling


def build_personalized_prompt(student_id, base_prompt):
    """Enhance prompt with student's learning profile"""
    profile = get_student_profile(student_id, None)
    
    personalization = f"""

STUDENT LEARNING PROFILE:
- Mastered concepts: {', '.join(profile.get('mastered_concepts', [])[:5]) or 'None yet'}
- Current struggles: {', '.join(profile.get('struggles', [])[:3]) or 'None identified'}
- Known misconceptions: {', '.join(profile.get('misconceptions', [])[:3]) or 'None identified'}
- Learning style: {profile.get('preferred_style', 'mixed')}
- Typical questions to understanding: {profile.get('avg_questions_to_understand', 5)}

ADAPT YOUR TEACHING:
- Build on what they already know (mastered concepts)
- Be patient with struggle areas
- Address known misconceptions when relevant
- Use their preferred learning style
"""
    
    return base_prompt + personalization

def auto_format_math(text):
    """Auto-detect and wrap math expressions that AI forgot to format"""
    import re
    
    # If already has lots of LaTeX, don't mess with it
    if text.count('$') > 2:
        return text
    
    # Only do basic fixes for common patterns
    
    # Fix: F = ma (simple equations)
    text = re.sub(
        r'\b([A-Z]_?[a-z0-9]*)\s*=\s*([A-Za-z0-9_*/^()+-]+)\b',
        r'$\1 = \2$',
        text
    )
    
    # Fix: F_c (subscripts)
    text = re.sub(
        r'\b([A-Z])_([a-z])\b',
        r'$\1_\2$',
        text
    )
    
    # Fix: m^2 (superscripts)
    text = re.sub(
        r'\b([a-z])\^(\d)\b',
        r'$\1^\2$',
        text
    )
    
    return text

# Add endpoint to change teaching mode
@app.route('/api/set_mode', methods=['POST', 'OPTIONS'])
def set_mode():
    if request.method == 'OPTIONS': return '', 200
    data = request.json or {}
    mode = data.get('mode', 'guided')
    if mode in ['socratic', 'guided', 'direct']:
        session['teaching_mode'] = mode
        return jsonify({"status": "success", "mode": mode})
    return jsonify({"status": "error", "message": "Invalid mode"}), 400

# Add endpoint for "explain this" button
@app.route('/api/explain', methods=['POST', 'OPTIONS'])
def explain():
    if request.method == 'OPTIONS': return '', 200
    data = request.json or {}
    topic = data.get('topic', '')
    subject = get_subject()
    
    prompt = f"""Explain {topic} in the context of {subject} in a clear, simple way.
    Use analogies and examples. Break it down into easy-to-understand steps.
    Keep it concise (3-4 sentences) but comprehensive."""
    
    genai_client = get_genai_client()
    model = genai_client.GenerativeModel(model_name)
    response = model.generate_content(prompt)    
    return jsonify({"answer": response.text})
# Math helpers
def _extract_json_block(text: str) -> str:
    if not text: return ""
    start = text.find('{'); end = text.rfind('}')
    return text[start:end+1] if start != -1 and end != -1 and end > start else text

def _get_decimal_places() -> int:
    try: return max(0, int(os.environ.get('MATH_DECIMALS','2')))
    except: return 2

def _normalize_numeric_string(value: float, places: int) -> str:
    s = f"{round(value, places):.{places}f}"
    return s.rstrip('0').rstrip('.') if '.' in s else s

def _generate_math_set(concept: str):
    prompt = f"""
    Create exactly 10 unique practice problems testing the concept: "{concept}"
    Rules:
    - The 10 problems must gradually increase in difficulty from 1 (easiest) to 10 (hardest).
    - Include a mix of numeric, word-based, and symbolic problems.
    - Output STRICT JSON: {{"problems":[{{"number":1,"question":"...","answer":"..."}},{{"number":10,"question":"...","answer":"..."}}]}}
    """
    client = get_genai_client()
    raw = client.models.generate_content(model=model_name, contents=prompt).text
    try:
        data = json.loads(_extract_json_block(raw))
        problems = data.get("problems", [])
    except:
        problems = []
    
    formatted, places = [], _get_decimal_places()
    for idx, p in enumerate(problems[:10], start=1):
        q = p.get("question") if isinstance(p, dict) else None
        a = p.get("answer") if isinstance(p, dict) else None
        if q and a is not None:
            try: 
                a_str = _normalize_numeric_string(float(str(a).replace(',','').strip()), places)
            except: 
                a_str = str(a).strip()
            formatted.append({"number": idx, "question": q, "answer": a_str})
    return formatted

def _parse_user_answers(text: str) -> dict:
    answers = {}
    for part in re.split(r"[\n;,]", text or ""):
        m = re.search(r"(\d{1,2})\s*[\)\:\-]?\s*(.+)", part.strip())
        if m: answers[int(m.group(1))] = m.group(2).strip()
    if not answers:
        tokens = [t for t in (text or "").replace("\n"," ").split(" ") if t.strip()]
        for i, tok in enumerate(tokens[:10], start=1): answers[i] = tok
    return answers

def _answers_equal(expected: str, given: str) -> bool:
    def to_num(s):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s) if s is not None else "")
        if m:
            try: return float(m.group(0).replace(",",""))
            except: return None
        return None
    en, gn = to_num(expected), to_num(given)
    if en is not None and gn is not None:
        places = _get_decimal_places()
        return _normalize_numeric_string(en, places) == _normalize_numeric_string(gn, places)
    return str(expected).strip() == str(given).strip()

def _generate_hint(question: str, correct_answer: str) -> str:
    try:
        client = get_genai_client()
        raw = client.models.generate_content(
            model=model_name,
            contents=f"Give a short hint without the final answer.\nQ: {question}\nA: {correct_answer}"
        ).text
        return (raw or "Check your setup carefully.").strip()
    except:
        return "Re-check your steps."

def _looks_like_answers(text: str) -> bool:
    if not text: return False
    if len(re.findall(r"\b(\d{1,2})\s*[\)\:\-]", text)) >= 3: return True
    return sum(text.count(sep) for sep in [',',';','\n']) >= 3

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/teacher/login')
def teacher_login_page():
    return render_template('teacher_login.html')

@app.route('/teacher/dashboard')
def teacher_dashboard():
    return render_template('teacher_dashboard.html')

@app.route('/health')
def health():
    return jsonify({"status":"ok"}), 200

@app.route('/api/set_subject', methods=['POST', 'OPTIONS'])
def set_subject_route():
    if request.method == 'OPTIONS': return '', 200
    data = request.json or {}
    subject = data.get('subject', '')
    set_subject(subject)
    
    if subject == 'Math':
        session['math_mode'] = {'problems': [], 'generated': False}
    else:
        session.pop('math_mode', None)
    
    return jsonify({"status": "success", "subject": subject})

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def ask():
    if request.method == 'OPTIONS': return '', 200
    data = request.json or {}
    user_input = data.get('message', '')
    subject = get_subject()
    
    # Get student and classroom from session (set during login)
    student_id = session.get('user_id')  # This is the student's UUID from Supabase
    classroom_id = session.get('classroom_id')
    
    # Get conversation history
    conversation = get_conversation()
    
    # === NEW: Detect struggle ===
    if subject and student_id and classroom_id:
        is_struggling = detect_struggle(student_id, classroom_id, subject, conversation)
        
        if is_struggling:
            # Add instruction to change approach
            session['ai_should_change_approach'] = True
    
    # Math mode (keep existing logic)
    if session.get('math_mode') is not None and subject == 'Math':
        math_state = session.get('math_mode') or {}
        
        # [Keep all your existing math mode logic here]
        # ... (unchanged)
        pass
    
    # === NEW: Build personalized prompt ===
    base_prompt = get_teaching_prompt(session.get('teaching_mode', 'guided'))
    
    if student_id:
        personalized_prompt = build_personalized_prompt(student_id, base_prompt)
        
        # Add struggle adaptation if detected
        if session.get('ai_should_change_approach'):
            personalized_prompt += """

⚠️ STRUGGLE DETECTED: Student has asked about this topic multiple times.
CHANGE YOUR STRATEGY:
1. Step back to check prerequisite understanding
2. Try a completely different explanation approach
3. Use Socratic questions to find the specific gap
4. Consider they may have a fundamental misconception
DO NOT repeat the same type of explanation."""
            session['ai_should_change_approach'] = False  # Reset
    else:
        personalized_prompt = base_prompt
    
    # Build full prompt with conversation history
    prompt = f"""{personalized_prompt}

Subject: {subject}
Student's message: "{user_input}"

Recent conversation: {conversation[-4:] if len(conversation) > 4 else conversation}

Your response (be concise, 3-4 sentences max):"""
    
    # Get AI response
    genai_client = get_genai_client()
    model = genai_client.GenerativeModel(model_name)
    response = model.generate_content(prompt)    
    ai_answer = response.text
    
    # Update conversation
    conversation.append({"role": "user", "content": user_input})
    conversation.append({"role": "ai", "content": ai_answer})
    session['conversation'] = conversation
    
    # === NEW: Update student profile periodically ===
    if len(conversation) % 6 == 0 and student_id:  # Every 3 exchanges
        try:
            update_student_profile_from_conversation(student_id, conversation)
        except Exception as e:
            logger.error(f"Error updating profile: {e}")
    
    return jsonify({"answer": ai_answer})


def get_teaching_prompt(teaching_mode):
    """Get base teaching prompt based on mode"""
    
    if teaching_mode == 'socratic':
        base = """You are Socrates, a master teacher who guides through questions.
        
CRITICAL RULES:
- NEVER solve homework problems directly
- Ask what they've tried first
- Guide them step-by-step to solve it themselves"""
        
    elif teaching_mode == 'direct':
        base = """You are a helpful tutor who explains concepts clearly.

CRITICAL RULES:
- NEVER solve homework problems directly
- Explain the METHOD and PROCESS clearly
- Give a DIFFERENT example (not their exact problem)"""
        
    else:  # guided (default)
        base = """You are an adaptive tutor who teaches effectively without doing the work for students.

CRITICAL RULES:
1. NEVER solve their homework problems directly
2. If it's a homework-like question: teach the method with a different example
3. If it's a concept question: give a clear explanation + follow-up question
4. If they're stuck: ask what specific part is confusing"""
    
    # ADD THIS LINE - append the formatting guide
    return base + "\n" + FORMATTING_GUIDE

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    session.clear()
    return jsonify({"ok": True})

@app.route('/api/teacher/summary', methods=['POST', 'OPTIONS'])
def teacher_summary():
    """Generate AI summary of classroom activity"""
    if request.method == 'OPTIONS': return '', 200
    
    data = request.get_json(silent=True) or {}
    messages = data.get('messages', [])
    classroom_name = data.get('classroom_name', 'this classroom')
    
    if not messages or len(messages) == 0:
        return jsonify({
            "ok": True, 
            "summary": "No student activity yet. Students haven't started asking questions."
        })
    
    # Limit to last 100 messages for performance
    recent_messages = messages[-100:] if len(messages) > 100 else messages
    
    # Build context for AI
    message_text = "\n".join([
        f"Student: {msg['content'][:200]}" 
        for msg in recent_messages 
        if msg.get('content')
    ])
    
    prompt = f"""You are analyzing student questions from {classroom_name} to help a teacher understand what's happening.

Student questions (last {len(recent_messages)} messages):
{message_text}

Generate a concise summary (2-3 paragraphs) covering:

1. **Main Topics**: What concepts are students asking about most?
2. **Learning Patterns**: Are students exploring deeply or asking surface-level questions?
3. **Potential Struggles**: What topics seem to confuse students? What misconceptions appear?
4. **Recommendations**: What should the teacher focus on in class?

Be specific, actionable, and encouraging. Focus on insights, not just listing topics.

Summary:"""
    
    try:
        genai_client = get_genai_client()
        model = genai_client.GenerativeModel(model_name)
        response = model.generate_content(prompt)        
        summary = response.text
        
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return jsonify({
            "ok": False, 
            "error": "Could not generate summary"
        }), 500
    
@app.route('/api/teacher/alerts', methods=['GET'])
@teacher_required
def get_teacher_alerts():
    """Get alerts for teacher's classrooms"""
    from supabase import create_client
    
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_ANON_KEY')
    supabase = create_client(supabase_url, supabase_key)
    
    teacher_id = session.get('user_id')
    classroom_id = request.args.get('classroom_id')
    
    if not classroom_id:
        return jsonify({"ok": False, "error": "classroom_id required"}), 400
    
    # Verify teacher owns this classroom
    classroom = supabase.table('classrooms').select('*').eq('id', classroom_id).eq('teacher_id', teacher_id).execute()
    if not classroom.data:
        return jsonify({"ok": False, "error": "Classroom not found"}), 404
    
    # Get unacknowledged alerts
    alerts = supabase.table('teacher_alerts').select('*, students(username)').eq('classroom_id', classroom_id).eq('acknowledged', False).order('created_at', desc=True).execute()
    
    # Format alerts
    formatted_alerts = []
    for alert in (alerts.data or []):
        formatted_alerts.append({
            'id': alert['id'],
            'student_name': alert['students']['username'] if alert.get('students') else 'Unknown',
            'student_id': alert['student_id'],
            'type': alert['alert_type'],
            'topic': alert['topic'],
            'severity': alert['severity'],
            'message': alert['message'],
            'recommendation': alert['recommendation'],
            'created_at': alert['created_at']
        })
    
    return jsonify({"ok": True, "alerts": formatted_alerts})


@app.route('/api/teacher/alerts/<alert_id>/acknowledge', methods=['POST'])
@teacher_required
def acknowledge_alert(alert_id):
    """Mark alert as acknowledged"""
    from supabase import create_client
    
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_ANON_KEY')
    supabase = create_client(supabase_url, supabase_key)
    
    supabase.table('teacher_alerts').update({'acknowledged': True}).eq('id', alert_id).execute()
    
    return jsonify({"ok": True})
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)