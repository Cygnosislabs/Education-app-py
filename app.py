from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import subprocess
import httpx

# Initialize Flask app
app = Flask(__name__)

# Mistral API Config
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# CORS for frontend
# Flask-CORS setup (equivalent to FastAPI's CORSMiddleware)
CORS(app)

# Load embedding model and vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Construct the path to the vectorstore relative to the current file
vectorstore_path = os.path.join(os.path.dirname(__file__), "./vectorstore")

# Load the FAISS vectorstore from the specified path
db = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)

# -------------------------------
# Utilities
# -------------------------------

def get_context(question: str, class_name: str, subject: str, lesson: str | None = None):
    """
    Retrieves relevant document context from the FAISS vectorstore based on the question
    and specified filters (class, subject, lesson).
    """
    filters = {
        "class": class_name,
        "subject": subject
    }
    if lesson:
        filters["lesson"] = lesson

    # Perform a similarity search in the FAISS database
    docs = db.similarity_search(question, k=1, filter=filters)
    # Join the page content of the retrieved documents into a single string
    return "\n\n".join([doc.page_content for doc in docs])

def get_lesson_context(class_name: str, subject: str, lesson: str | None = None):
    """
    Retrieves general lesson content from the FAISS vectorstore based on filters.
    This is used for revision, where a specific question's similarity is not the primary concern.
    It fetches a few top documents related to the lesson.
    """
    filters = {
        "class": class_name,
        "subject": subject
    }
    if lesson:
        filters["lesson"] = lesson

    # Use a general query to fetch top k documents with scores
    docs_and_scores = db.similarity_search_with_score("", k=1, filter=filters)
    relevant_docs = [doc for doc, score in docs_and_scores if score > 0.85]

    if not relevant_docs:
        return "Sorry, I couldn't find anything relevant to the current topic."

    # Join the page content of relevant docs
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context


def run_ollama(prompt: str):
    """
    Executes a command to run a Mistral model via Ollama.
    This function is kept from the original code but is not used in the current
    Mistral API integration. It's here for completeness if Ollama is preferred.
    """
    result = subprocess.run(
        ['ollama', 'run', 'mistral', prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    if result.returncode != 0:
        # If Ollama command fails, abort with a 500 HTTP error
        abort(500, description=f"Ollama error: {result.stderr.strip()}")
    return result.stdout.strip()

def run_mistral(prompt: str) -> str:
    """
    Sends a prompt to the Mistral AI API and returns the generated response.
    Handles API authorization, payload construction, and error handling.
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small", # Using 'mistral-small' as specified in the original code
        "messages": [
            {"role": "system", "content": "You are a helpful and friendly educational assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7, # Controls the randomness of the output
        "max_tokens": 512 # Maximum number of tokens to generate in the response
    }

    try:
        # Make a POST request to the Mistral API
        response = httpx.post(MISTRAL_API_URL, json=payload, headers=headers)
        response.raise_for_status() # Raise an HTTPStatusError for bad responses (4xx or 5xx)
        # Extract and return the content of the AI's message
        return response.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        # Log and abort with a 500 error if there's an HTTP status error from Mistral
        print("Status code:", e.response.status_code)
        print("Response body:", e.response.text)
        abort(500, description=f"Mistral API error: {e.response.text}")
    except Exception as e:
        # Log and abort with a 500 error for any other unexpected exceptions
        print("General exception:", str(e))
        abort(500, description=f"Unexpected error: {str(e)}")

# -------------------------------
# Routes
# -------------------------------
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    if not data:
        abort(400, description="Invalid JSON data provided.")

    question = data.get("question")
    class_name = data.get("class_name")
    subject = data.get("subject")
    lesson = data.get("lesson")

    if not all([question, class_name, subject]):
        abort(400, description="Missing required fields: question, class_name, or subject.")

    # Get context
    context = get_context(question, class_name, subject, lesson)

    if context is None:
        return jsonify({
            "response": "Sorry, I couldn't find anything relevant to your question in the current topic."
        })

    prompt = f"""You are an educational assistant helping students.
Only answer if the context is related to the question. If not, say "Sorry, I don't have enough information to answer that."

LESSON:
{context}

QUESTION:
{question}

Answer:"""

    return jsonify({"response": run_mistral(prompt)})


@app.route("/generate", methods=["POST"])
def generate_questions():
    """
    API endpoint to generate comprehension questions based on lesson context.
    Expects JSON input with 'question', 'class_name', 'subject', and optional 'lesson'.
    """
    # Get JSON data from the request body
    data = request.json
    if not data:
        abort(400, description="Invalid JSON data provided.")

    # Extract required fields
    question = data.get("question")
    class_name = data.get("class_name")
    subject = data.get("subject")
    lesson = data.get("lesson")

    # Validate required fields
    if not all([question, class_name, subject]):
        abort(400, description="Missing required fields: question, class_name, or subject.")

    # Get context from the vectorstore
    context = get_context(question, class_name, subject, lesson)
    # Construct the prompt for generating questions
    prompt = f"""Generate 5 short, student-friendly comprehension questions based on the lesson below:

LESSON:
{context}

QUESTIONS:"""
    # Get the generated questions from the Mistral model and return as JSON
    return jsonify({"questions": run_mistral(prompt)})

@app.route("/feedback", methods=["POST"])
def give_feedback():
    """
    API endpoint to provide feedback on a student's answer by comparing it to the correct answer
    derived from the lesson context.
    Expects JSON input with 'question', 'student_answer', 'class_name', 'subject', and optional 'lesson'.
    """
    # Get JSON data from the request body
    data = request.json
    if not data:
        abort(400, description="Invalid JSON data provided.")

    # Extract required fields
    question = data.get("question")
    student_answer = data.get("student_answer")
    class_name = data.get("class_name")
    subject = data.get("subject")
    lesson = data.get("lesson")

    # Validate required fields
    if not all([question, student_answer, class_name, subject]):
        abort(400, description="Missing required fields: question, student_answer, class_name, or subject.")

    # Get context from the vectorstore
    context = get_context(question, class_name, subject, lesson)
    # Construct the prompt for generating feedback
    prompt = f"""Use the lesson below to answer the question, then evaluate the student's answer.

LESSON:
{context}

QUESTION:
{question}

STUDENT ANSWER:
{student_answer}

INSTRUCTIONS:
1. Provide the correct answer.
2. Compare it with the student's answer.
3. Give clear and constructive feedback.

RESPONSE:"""
    # Get the feedback from the Mistral model and return as JSON
    return jsonify({"feedback": run_mistral(prompt)})

@app.route("/revise", methods=["POST"])
def revise_lesson():
    """
    API endpoint to provide a detailed, story-like explanation of a lesson for revision.
    Expects JSON input with 'class_name', 'subject', and optional 'lesson'.
    """
    # Get JSON data from the request body
    data = request.json
    if not data:
        abort(400, description="Invalid JSON data provided.")

    # Extract required fields
    class_name = data.get("class_name")
    subject = data.get("subject")
    lesson = data.get("lesson")

    # Validate required fields
    if not all([class_name, subject]):
        abort(400, description="Missing required fields: class_name or subject.")

    # Get general lesson context (not question-specific)
    context = get_lesson_context(class_name, subject, lesson)
    # Construct the prompt for lesson revision/explanation
    prompt = f"""You are a school teacher explaining a lesson to students in a detailed, story-like manner.
Start with the lesson name: {lesson} as the first line — without using any symbols like asterisks.
Then, explain the full lesson in a simple, engaging way so that students can clearly understand it.
Use storytelling style, real-life comparisons if needed. Do not use bullet points. Do not say it's a revision or explanation. Just start teaching like in a classroom.

Here is the LESSON content:
{context}

Begin teaching now:"""
    # Get the revised lesson explanation from the Mistral model and return as JSON
    return jsonify({"response": run_mistral(prompt)})

from flask import send_from_directory

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    return send_from_directory("frontend", path)

# Entry point for running the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

