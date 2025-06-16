# 🤗 Anita: AI Therapy Chatbot

Anita is a compassionate AI-powered therapy chatbot built using BERT-based intent classification, semantic search with FAISS, and LLM-driven responses via Gemini Pro. It offers emotionally aware conversations and generates speech output using Sarvam AI’s multilingual TTS API.

## Important Note:
This project is for learning and experimentation only.
It is not intended to replace real therapy or mental health professionals in any manner. If you’re struggling emotionally, always speak to a qualified therapist or counselor.

---

## Features

- **Intent Classification**: Uses a fine-tuned BERT model to detect emotional intent.
- **Therapeutic Context Retrieval**: Retrieves real-world therapy conversations using semantic search (FAISS + Sentence Transformers).
- **LLM Response Generation**: Generates deeply empathetic replies using Gemini-Pro LLM.
- **Sarvam TTS Integration**: Converts AI responses to natural-sounding audio in Indian languages and voices.
- **Chat Memory**: Stores and reloads conversations with session memory.
- **Customizable Voice & Accent**: Choose from 10+ languages and 6 speaker voices.

---

## Getting Started

### 1. Prerequisites:
  a. 3.11 ≤ Python < 3.12.5 with Conda <br/>
  b. [GOOGLE API KEY](https://aistudio.google.com/app/apikey) <br/>
  c. [SARVAM API KEY](https://dashboard.sarvam.ai/) <br/>
  
### 2. Clone the Repository

```
  git clone https://github.com/CBhandawat/Anita-Therapy-ChatBot.git
  cd Anita-Therapy-Chatbot
```

### 3. Setup
   Create Virtual Environment:
     ```
     python -m venv .venv
     ```

   Activate:
     ```
     .\.venv\Scripts\activate
     ```

   Install all requirements:
     ```
     pip install -r requirements.txt
     ```
### 4. Set Environment Variables
  Create .env file at the root: <br/>
    ```
    GOOGLE_API_KEY = "YOUR API KEY HERE"
    SARVAM_API_KEY = "YOUR API KEY HERE"
    ```
    
### 5. Run
   ```
   streamlit run app.py
   ```
It will open your browser

# LICENSE
[Apache 2.0 License](https://github.com/CBhandawat/Anita_Therapy_ChatBot/blob/main/LICENSE)

