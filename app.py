# Useful link: https://seayeshaiftikhar.medium.com/customer-support-chatbot-using-python-168e0a7c958d

# Import libraries
from flask import Flask, render_template, request
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import numpy as np
import pickle
import os
import requests
from dotenv import load_dotenv
import re

# Download NLTK resources
nltk.download('popular')

# Initialize lemmatizer and load saved resources
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('data/data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Flask App Initialization
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the API key stored in the file "api.env"
load_dotenv("api.env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# File where questions and answers will be saved
CACHE_FILE = "cache.json"

# Load the cache from file if it exists
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

# Save updated cache back to file
def save_cache(cache):
    with open(CACHE_FILE, "w") as file:
        json.dump(cache, file)

# Load the cache file at the start of app
cache = load_cache()

# Utility Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model, threshold=0.25):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(msg):
    try:
        ints = predict_class(msg, model)
        return get_response(ints, intents)
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, something went wrong."

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get("msg")

    if not OPENROUTER_API_KEY:
        return "Error: Missing API Key"

    try:
        user_text_lower = user_text.lower().strip()

        # Check if the response is already cached
        if user_text_lower in cache:
            return cache[user_text_lower]

        # Simple direct responses
        if any(greet in user_text_lower for greet in ["hey", "hello", "hi there", "good morning", "good evening", "good afternoon"]):
            return "Hello! I'm here to support you. What's on your mind today?"

        if "how are you" in user_text_lower:
            return "I'm good and ready to support you. How are you feeling today?"

        if "what can you do" in user_text_lower or "what are you" in user_text_lower:
            return "I'm MindEaseAI, always here to support you whenever you're going through a tough time."

        if "who are you" in user_text_lower or "what are you" in user_text_lower:
            return "I'm MindEaseAI, a mental health assistant created to offer calm, caring support whenever you need it."

        restricted_topics = ["history", "gaming", "politics", "finance"]
        if any(topic in user_text_lower for topic in restricted_topics):
            return "I'm here to support your mental well-being. Let's talk about stress, self-care, mindfulness, or emotions."

        # Prepare API Call
        API_URL = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        # Data to send to the GPT model
        data = {
            "model": "mistralai/mistral-small-3.1-24b-instruct:free",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are MindEaseAI, a kind and calming mental health assistant. "
                        "Provide supportive suggestions in a gentle, natural tone. "
                        "If the user greets you (e.g., says 'hi', 'hello', or similar), reply with a short friendly greeting. "
                        "If mentioned stress or any health issue suggest a few exercises and motivate the person. "
                        "Avoid numbered steps or disclaimers. Do not mention US-specific services. "
                        "If the user's message suggests a serious health crisis, respond gently but clearly: If you are in immediate danger, please call 911. "
                        "Use ✧ to introduce each suggestion. Add gentle emojis like 😇, 💙, 🌿, 🌟, 💪. "
                        "Make your replies emotionally warm, clean, and helpful."
                    )
                },
                {"role": "user", "content": user_text}
            ],
            "max_tokens": 500
        }

        # Send the request
        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            response_json = response.json()
            if "choices" in response_json:
                reply = response_json["choices"][0]["message"]["content"]

                # Clean the reply text
                reply = re.sub(r"(?i)(i'?m (really )?sorry.*?help.*?)\.", "You're not alone in this. Let's try something helpful.", reply)
                reply = re.sub(r"(if.*?suicidal.*?call.*?)\.", "If you are in immediate danger, please call 911.", reply, flags=re.IGNORECASE)
                reply = re.sub(r"talk suicide canada.*?\.", "If you are in immediate danger, please call 911.", reply, flags=re.IGNORECASE)
                reply = re.sub(r'\b\d\.\s*', '', reply)
                reply = reply.replace('\n', ' ')
                formatted_reply = re.sub(r'(?<=[.!?])\s+', '<br>', reply.strip())

                # Save the new question and answer into cache
                cache[user_text_lower] = formatted_reply
                save_cache(cache)

                return formatted_reply
            else:
                return f"Error: {response_json}"
        else:
            return f"Error: {response.json()}"

    except Exception as e:
        return f"Error: {str(e)}"

# Main Execution
if __name__ == "__main__":
    app.run(debug=True)
