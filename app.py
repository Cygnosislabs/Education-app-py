from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import subprocess
import httpx
import requests

# Initialize Flask app
app = Flask(__name__)

# Mistral API Config
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "rbRjkQ3xplI3IZjbjWnUzwZfLaDC610z"

# Google Places API
GOOGLE_PLACES_API_KEY = "AIzaSyC5S2Uo2TfxWjvy-_8klbPCzJmkcsUB_VA"

# CORS for frontend
CORS(app)

# Load embedding model and vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore_path = os.path.join(os.path.dirname(__file__), "./vectorstore")

db = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)


# -------------------------
# AI + Helper Functions
# -------------------------

def run_ollama(prompt: str):
    result = subprocess.run(
        ['ollama', 'run', 'mistral', prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    if result.returncode != 0:
        abort(500, description=f"Ollama error: {result.stderr.strip()}")
    return result.stdout.strip()


def run_mistral(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small",
        "messages": [
            {"role": "system", "content": "You are a helpful and medical assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        response = httpx.post(MISTRAL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        abort(500, description=f"Mistral API error: {e.response.text}")
    except Exception as e:
        abort(500, description=f"Unexpected error: {str(e)}")


# -------------------------
# CATEGORY DETECTION
# -------------------------

CATEGORY_KEYWORDS = {
    "skin": "dermatologist",
    "hair": "trichologist",
    "dentist": "dentist",
    "eye": "eye hospital",
    "ent": "ENT specialist",
    "general": "general physician"
}


def classify_issue(query: str) -> str:
    prompt = f"""
Classify the user's health problem into one of these:
skin, hair, dentist, eye, ent, general.

Only reply one category without explanation.

User query: {query}
"""
    category = run_mistral(prompt).lower().strip()
    return category if category in CATEGORY_KEYWORDS else "general"


# -------------------------
# GOOGLE PLACES SEARCH
# -------------------------

def get_nearby_hospitals(lat, lng, keyword):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": 3000,
        "keyword": keyword,
        "key": GOOGLE_PLACES_API_KEY
    }

    response = requests.get(url, params=params).json()

    hospitals = []
    for place in response.get("results", []):
        hospitals.append({
            "name": place.get("name"),
            "address": place.get("vicinity"),
            "rating": place.get("rating", "N/A"),
            "lat": place["geometry"]["location"]["lat"],
            "lng": place["geometry"]["location"]["lng"]
        })

    return hospitals


# -------------------------
# ROUTES
# -------------------------

@app.route("/askDoctor", methods=["POST"])
def ask_doctor():
    data = request.json
    if not data:
        abort(400, description="Invalid JSON data provided.")

    question = data.get("question")

    if not question:
        abort(400, description="Missing required field: question.")

    prompt = f"""
You are a medical assistant helping users with general health information.
Provide clear, safe, and friendly responses.
Do not give diagnoses or medical treatment.
Always recommend consulting a licensed doctor.

QUESTION:
{question}

Answer:
"""

    return jsonify({"response": run_mistral(prompt)})


# -------------------------
# NEW: Nearby Hospitals API
# -------------------------

@app.route("/nearbyHospitals", methods=["POST"])
def nearby_hospitals():
    data = request.json

    if not data:
        abort(400, "Invalid JSON data.")

    query = data.get("query")
    lat = data.get("lat")
    lng = data.get("lng")

    if not query:
        abort(400, "Missing field: query.")
    if lat is None or lng is None:
        abort(400, "Missing location (lat, lng).")

    # 1) Classify medical category
    category = classify_issue(query)
    keyword = CATEGORY_KEYWORDS.get(category, "hospital")

    # 2) Get nearby hospitals
    hospitals = get_nearby_hospitals(lat, lng, keyword)

    # 3) Return structured response
    return jsonify({
        "category": category,
        "search_keyword": keyword,
        "hospitals": hospitals
    })


# -------------------------
# ENTRY POINT
# -------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

