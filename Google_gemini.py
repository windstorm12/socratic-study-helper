from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

# ----------------------
# Configure AI
# ----------------------
genai.configure(api_key="AIzaSyBRZdFOc5GljNkEoZdLh-HA_EjfM3q_kRA")
model = genai.GenerativeModel("gemini-1.5-flash")

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
    the learning gaps the user has. If you see any incorrect explaination your question should indirectly convey that the user might be incorrect.
    If you notice any incorrect explainations, ask the user some critical thinking questions that challenge their understanding. if the user 
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
    response = model.generate_content(prompt)
    ai_answer = response.text
    AI_append(ai_answer)
    return ai_answer

# ----------------------
# Flask app
# ----------------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Flask will look inside 'templates' folder

# Route to set the subject
@app.route('/set_subject', methods=['POST'])
def set_subject():
    global user_subject
    data = request.json
    user_subject = data.get('subject', '')
    return jsonify({"status": "success", "subject": user_subject})

# Route to ask a question
@app.route('/ask', methods=['POST'])
def ask():
    global user_subject
    data = request.json
    user_input = data.get('message', '')

    # Append user input
    User_append(user_input)

    # Get AI response
    ai_answer = get_answer(user_subject, user_input)

    return jsonify({"answer": ai_answer})

# Run server
if __name__ == "__main__":
    app.run(debug=True)
