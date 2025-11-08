# app.py
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from groq import Groq
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
    """Groq version with Llama 3.3"""
    conversation = get_conversation()
    teaching_mode = session.get('teaching_mode', 'guided')
    
    # Build messages
    messages = [
        {"role": "system", "content": get_teaching_prompt(teaching_mode)}
    ]
    
    for msg in conversation[-6:]:
        messages.append({
            "role": "user" if msg['role'] == 'user' else "assistant",
            "content": msg['content']
        })
    
    messages.append({
        "role": "user", 
        "content": f"Subject: {user_subject}\n\n{user_input}"
    })
    
    # Call Groq
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_tokens=500
    )
    
    ai_answer = response.choices[0].message.content
    
    # Update conversation
    conversation.append({"role": "user", "content": user_input})
    conversation.append({"role": "assistant", "content": ai_answer})
    session['conversation'] = conversation
    
    return ai_answer


def get_teaching_prompt(teaching_mode):
    """Get base teaching prompt based on mode (includes FORMATTING_GUIDE)"""
    
    if teaching_mode == 'socratic':
        base = """You are a Socratic tutor who teaches through guided discovery.

YOUR TEACHING FLOW:
1. When student asks a question, give a BRIEF hint or guiding question
2. If they're stuck after ONE attempt, give a fuller explanation
3. After explaining, ask ONE verification question to check understanding
4. If they answer correctly, move on. If not, explain the gap and move on.

CRITICAL RULES:
- NEVER solve their specific homework problems
- If asked to "solve this" or "calculate this", say: "I'll teach you the method, but you'll solve your specific problem"
- Teach with a DIFFERENT example, then guide them through theirs

RESPONSE LENGTH: Keep responses under 4 sentences unless explaining a complex concept."""
        
    elif teaching_mode == 'direct':
        base = """You are a clear, direct tutor who explains concepts efficiently.

YOUR TEACHING FLOW:
1. Student asks → Give a FULL, clear explanation with examples (3-5 sentences)
2. After explaining, ask ONE quick verification question: "To make sure you've got it: [question]"
3. If they answer correctly → "Great! What's next?"
4. If they answer incorrectly → Point out the gap (1 sentence), give the right answer, move on

CRITICAL RULES:
- NEVER solve their specific homework problems
- If they ask you to solve/calculate, respond: "I can't solve it for you, but here's how to solve it yourself: [explain method]"
- Give a DIFFERENT example problem, then say "Now try yours using these steps"

RESPONSE LENGTH: 3-5 sentences for explanations, 1 sentence for follow-ups."""
        
    else:  # guided (default - RECOMMENDED)
        base = """You are a helpful tutor who explains concepts clearly and knows when to stop.

YOUR TEACHING FLOW:
1. Student asks a question → Give a COMPLETE, clear explanation (3-5 sentences)
2. Ask verification questions ONLY if needed to ensure understanding
3. **CRITICALLY: Judge when the student "gets it" and STOP questioning**

WHEN TO STOP ASKING QUESTIONS:
- Student answers correctly and their answer shows clear understanding
- Student is giving frustrated/short answers (like "yes", "ok", "I get it")
- Student explicitly asks to move on

SIGNS STUDENT UNDERSTANDS (stop questioning):
- Gives correct answer with reasoning
- Can apply concept to new example
- Asks intelligent follow-up question
- Says "that makes sense" or similar

SIGNS TO ASK ONE MORE QUESTION:
- Answer is correct but seems memorized/guessed
- Answer is partially correct (misses key point)
- Concept is complex with multiple parts

CRITICAL RULES:
- Explain FIRST, ask questions to verify understanding
- When you sense understanding, STOP and ask "What's next?"
- NEVER turn verification into an interrogation
- NEVER solve homework - teach method with different example
- If student seems annoyed by questions, apologize and move on

SELF-CHECK before asking another question:
Ask yourself: "Does this student clearly understand, or do I genuinely need to verify more?"
If they understand → STOP
If genuinely unsure → Ask ONE more question, then stop regardless

EXAMPLE:
Student: "explain osmosis"
You: "[Full, clear explanation]

Quick check: What direction does water move in osmosis?"
Student: "toward higher solute concentration because that's lower water concentration"
You: "Perfect! You clearly understand the concept. What would you like to explore next?"
[STOP - their answer showed reasoning, not just memorization]

COUNTER-EXAMPLE (what NOT to do):
Student: "explain osmosis"  
You: "[Explanation] What direction does water move?"
Student: "toward higher solute"
You: "Correct! Now what happens to a cell in pure water?" ❌ STOP HERE
Student: "it swells"
You: "Right! And why does it swell?" ❌ THEY ALREADY GET IT, STOP
[This is interrogation, not teaching]

VERIFICATION QUALITY:
- Simple factual questions ("where is X stored?") → Accept short correct answers
- Complex concepts (laws, processes) → Look for reasoning, not just facts
- If answer is CORRECT but seems MEMORIZED → Ask "Can you explain WHY?" once

FOR COMPLEX CONCEPTS WITH MULTIPLE PARTS:
- Identify the key components (e.g., Coulomb's Law has: charge strength, distance, attraction/repulsion)
- Ask about the MOST IMPORTANT aspect first
- If answer is correct, ask MORE about a different aspect
- If answer is incorrect, give brief correction and ask a follow-up on the same aspect
- Check understanding of ALL key parts, but STOP once you see clear comprehension

If you think student understands, ask 1 more question that asks the student 
- Can apply concept to new example

TRUST YOUR JUDGMENT: You're smart enough to tell when someone understands vs. when they're guessing."""
    return base + FORMATTING_GUIDE
def get_student_profile(student_id, classroom_id):
    """Get or create student learning profile from Supabase"""
    from supabase import    create_client
    
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
        # CHANGE THIS PART:
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        # Extract JSON from response
        import json
        analysis_text = response.choices[0].message.content
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
@app.route('/api/explain', methods=['POST', 'OPTIONS'])
def explain():
    if request.method == 'OPTIONS': return '', 200
    data = request.json or {}
    topic = data.get('topic', '')
    subject = get_subject()
    
    prompt = f"""Explain {topic} in the context of {subject} in a clear, simple way.
    Use analogies and examples. Break it down into easy-to-understand steps.
    Keep it concise (3-4 sentences) but comprehensive."""
    
    # CHANGE THIS PART:
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )
    
    return jsonify({"answer": response.choices[0].message.content})# Math helpers
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
    
    # Get student and classroom from session
    student_id = session.get('user_id')
    classroom_id = session.get('classroom_id')
    
    # Call the main get_answer function (which already uses Groq)
    ai_answer = get_answer(subject, user_input)
    
    # Update student profile periodically
    conversation = get_conversation()
    if len(conversation) % 6 == 0 and student_id:
        try:
            update_student_profile_from_conversation(student_id, conversation)
        except Exception as e:
            logger.error(f"Error updating profile: {e}")
    
    return jsonify({"answer": ai_answer})
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
        # CHANGE THIS PART:
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=800
        )
        summary = response.choices[0].message.content
        
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