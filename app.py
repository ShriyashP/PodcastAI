"""
Podcast Q&A service with audio transcription and question answering.

This FastAPI application demonstrates how you might build an end‑to‑end pipeline
for turning podcast audio into a searchable transcript and answering user
questions.  It includes endpoints for uploading audio files or providing a
direct link to an audio resource.  The audio is transcribed using OpenAI's
Whisper model via the `openai` Python library.  After transcription, the text
is split into sentences, vectorised with TF‑IDF and cached so that question
answering is fast.  A simple similarity‑based approach extracts the most
relevant sentence as the answer.
"""

from __future__ import annotations

import io
import re
from typing import List, Dict, Tuple, Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import openai

openai.api_key = ""

# FastAPI app and templates directory
app = FastAPI(title="Podcast QA Service", version="0.1.0")
templates = Jinja2Templates(directory=str((__file__).rsplit("/", 1)[0] + "/templates"))

# State to hold the latest transcript and its vectorisation
class TranscriptStore:
    """Holds a transcript and its TF‑IDF representation for QA."""

    def __init__(self):
        self.transcript: str = ""
        self.sentences: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def update(self, text: str):
        """Update the store with a new transcript."""
        self.transcript = text
        # Split into sentences
        self.sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        # Fit TF‑IDF vectoriser on sentences
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.matrix = self.vectorizer.fit_transform(self.sentences)

    def answer(self, question: str) -> str:
        """Return the sentence from the transcript that best matches the question."""
        if not self.sentences or self.vectorizer is None or self.matrix is None:
            return "No transcript available."
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        if sims.size == 0:
            return "Sorry, I don't know."
        best_idx = sims.argmax()
        return self.sentences[best_idx]


transcript_store = TranscriptStore()

# Helper functions

def clean_text(text: str) -> List[str]:
    """Tokenise text and filter stopwords for tag generation."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text)
    filtered = [t.lower() for t in tokens if t.lower() not in ENGLISH_STOP_WORDS]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(5)]


async def transcribe_bytes(audio_bytes: bytes) -> str:
    """Transcribe raw audio bytes using OpenAI Whisper."""
    if not openai.api_key:
        raise HTTPException(status_code=503, detail="OpenAI API key missing.")
    # Wrap the bytes in a BytesIO object
    file_obj = io.BytesIO(audio_bytes)
    try:
        transcription = openai.Audio.transcribe("whisper-1", file=file_obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    return transcription["text"]


async def download_audio(url: str) -> bytes:
    """Download audio content from a URL."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {e}")
    return resp.content

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the main page with upload and question forms."""
    return templates.TemplateResponse("index.html", {"request": request, "transcript": transcript_store.transcript})


@app.post("/transcribe_file", response_class=JSONResponse)
async def transcribe_file(file: UploadFile = File(...)) -> JSONResponse:
    """Receive an uploaded audio file, transcribe it and store the result."""
    audio_bytes = await file.read()
    text = await transcribe_bytes(audio_bytes)
    transcript_store.update(text)
    tags = clean_text(text)
    return JSONResponse({"transcript": text, "tags": tags})


@app.post("/transcribe_url", response_class=JSONResponse)
async def transcribe_url(url: str = Form(...)) -> JSONResponse:
    """Download audio from a URL, transcribe it and store the result."""
    audio_bytes = await download_audio(url)
    text = await transcribe_bytes(audio_bytes)
    transcript_store.update(text)
    tags = clean_text(text)
    return JSONResponse({"transcript": text, "tags": tags})


@app.post("/ask", response_class=JSONResponse)
async def ask_question(question: str = Form(...)) -> JSONResponse:
    """Answer a question based on the current transcript."""
    answer = transcript_store.answer(question)
    return JSONResponse({"answer": answer})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)