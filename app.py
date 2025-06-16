import streamlit as st
import torch
import faiss
import json
import pandas as pd
import os
import time
import joblib
import base64
import requests
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.set_page_config(page_title="Anita: Therapy Chatbot", page_icon="🤗")
st.title("🤗 Anita Therapy Chatbot")
st.markdown("Your AI therapist, trained on real-world therapy conversations and powered by empathy.")

# ------------------ Load Models Safely ------------------

def safe_load_model():
    try:
        model = BertForSequenceClassification.from_pretrained("model/hf_intent")
        tokenizer = BertTokenizer.from_pretrained("model/hf_intent")
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load BERT intent model: {e}")
        return None, None

intent_model, intent_tokenizer = safe_load_model()

try:
    with open("model/intent_id_map.json") as f:
        id_map = json.load(f)
    id2label = {int(k): v for k, v in id_map["id2label"].items()}
except Exception as e:
    st.error(f"Failed to load intent mapping: {e}")
    id2label = {}

try:
    with open("data/intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)["intents"]
    intent_map = {i["tag"]: i["responses"] for i in intents}
except Exception as e:
    st.error(f"Failed to load intents.json: {e}")
    intent_map = {}

try:
    index = faiss.read_index("model/faiss_index/therapy.index")
    df = pd.read_csv("model/faiss_index/data.csv")
except Exception as e:
    st.error(f"Error loading FAISS or CSV data: {e}")
    index, df = None, pd.DataFrame()

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to load embedding model: {e}")
    embedder = None

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or "")
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    model = None

# ------------------ Functions ------------------

def classify_intent(text):
    if not intent_model or not intent_tokenizer:
        return "unknown"
    try:
        inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = intent_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return id2label.get(pred, "unknown")
    except Exception as e:
        st.error(f"Intent classification error: {e}")
        return "unknown"

def get_therapy_response(query, k=1):
    if not index or df.empty or not embedder:
        return "I'm here to listen, please share more."
    try:
        embedding = embedder.encode([query])
        distances, indices = index.search(embedding, k)
        return df.iloc[indices[0][0]]["completion"]
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return "Thank you for sharing. Please continue."

def speak_text(text):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        st.warning("Sarvam API Key not found.")
        return

    try:
        def split_text(text, max_length=500, max_parts=3):
            words = text.split()
            chunks, current = [], ""
            for word in words:
                if len(current) + len(word) + 1 <= max_length:
                    current += (" " + word if current else word)
                else:
                    chunks.append(current)
                    current = word
                    if len(chunks) == max_parts - 1:
                        remaining = " ".join(words[words.index(word):])
                        chunks.append(remaining[:max_length].rsplit(' ', 1)[0])
                        break
            if current and len(chunks) < max_parts:
                chunks.append(current)
            return chunks

        text_chunks = split_text(text)

        payload = {
            "speaker": st.session_state.get('selected_speaker', 'meera'),
            "target_language_code": st.session_state.get('selected_language', 'hi-IN'),
            "inputs": text_chunks,
            "pitch": 1,
            "pace": 1,
            "loudness": 1,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": "bulbul:v1"
        }

        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json"
        }

        response = requests.post("https://api.sarvam.ai/text-to-speech", json=payload, headers=headers)

        if response.status_code == 200:
            audio_b64_list = response.json().get("audios", [])
            if not audio_b64_list:
                st.warning("No audio data received.")
                return

            full_audio_bytes = b''.join([base64.b64decode(b64) for b64 in audio_b64_list])
            b64_encoded_audio = base64.b64encode(full_audio_bytes).decode()

            audio_html = f"""
                <audio autoplay hidden>
                    <source src="data:audio/wav;base64,{b64_encoded_audio}" type="audio/wav">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.error(f"Sarvam TTS API Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Text-to-Speech error: {e}")

def generate_response_with_llm(user_input, intent, faiss_example, history):
    if not model:
        return "I'm unable to respond right now. Please try again later."

    try:
        system_prompt = """
You are Anita, a compassionate and thoughtful AI therapist.

Your goal is to deeply understand the user's emotional state over a multi-turn conversation. You should not give quick conclusions or generic advice. Ask meaningful follow-up questions, reflect on what the user says, and guide the user into exploring their feelings more deeply.

Never repeat the same question. Never give a flat or generic answer like "Tell me more". Always build upon what was said earlier.

When enough information is gathered, summarize key emotions and offer thoughtful suggestions. End by gently recommending that they talk to a professional human therapist if needed.

Keep your responses as short as possible and in simple English

Use this structure:
- Gently reflect their emotions
- Ask a related follow-up
- Stay on topic and deepen understanding
- Never jump to solutions too early
        """

        history_text = ""
        for msg in history[-6:]:
            history_text += f"User: {msg['user']}\nAnita: {msg['anita']}\n"

        prompt = f"""
{system_prompt}

Conversation so far:
{history_text}

Detected intent: {intent}
User's latest message: {user_input}

A similar real therapy case for inspiration:
{faiss_example}

Now, write Anita's next reply. It should feel like a caring, smart continuation of the above.
        """

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"LLM generation failed: {e}")
        return "I'm here to support you. Please continue."

# ------------------ Streamlit Chat UI with Memory ------------------

# Sidebar options for language and speaker
st.sidebar.title("Choose Your Voice")

language_options = ['en-IN', 'hi-IN', 'bn-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'gu-IN']
speaker_options = ['meera', 'pavithra', 'maitreyi', 'arvind', 'amol', 'amartya']

selected_language = st.sidebar.selectbox("Select Accent", language_options, index=0)
selected_speaker = st.sidebar.selectbox("Select Voice", speaker_options, index=0)

# Save to session
st.session_state['selected_language'] = selected_language
st.session_state['selected_speaker'] = selected_speaker


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = f"chat-{int(time.time())}"

chat_file = f"data/{st.session_state.chat_id}-messages"

# Load previous chat from disk
if os.path.exists(chat_file):
    st.session_state.messages = joblib.load(chat_file)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["anita"])

# Chat input
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # Save user message
    st.session_state.messages.append({"user": user_input, "anita": "..."})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Anita is listening..."):
        intent = classify_intent(user_input)
        faiss_response = get_therapy_response(user_input)
        final_response = generate_response_with_llm(
            user_input=user_input,
            intent=intent,
            faiss_example=faiss_response,
            history=st.session_state.messages  
        )
        speak_text(final_response)

    st.session_state.messages[-1]["anita"] = final_response

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Save chat to disk
    os.makedirs("data", exist_ok=True)
    joblib.dump(st.session_state.messages, chat_file)
