"""
config.py
=========
This file is the single source of truth for every "tunable" value in the
application: file paths, chunking parameters, model names, etc.

WHY THIS FILE EXISTS
---------------------
Without a config file, "magic numbers" (like chunk_size=1000) end up
copy-pasted across many files. If you ever want to experiment with a
different chunk size or swap the LLM model, you'd have to hunt through
every file to change it. By keeping everything here, every other module
just does:

    from config import CHUNK_SIZE

...and the whole app updates consistently.

We use Python's built-in `os` module to build file paths that work on
Windows, Mac, and Linux without modification (no hardcoded slashes).
"""

import os
from dotenv import load_dotenv

# load_dotenv() reads the `.env` file (if present) in the project root and
# copies its key=value pairs into the process's environment variables
# (accessible later via os.environ / os.getenv). This is how we keep secrets
# like API keys OUT of source code.
load_dotenv()

# ---------------------------------------------------------------------------
# BASE PATHS
# ---------------------------------------------------------------------------
# os.path.dirname(os.path.abspath(__file__)) gives the absolute folder that
# this config.py file lives in. Every other path is built relative to this,
# so the app works no matter where you clone/run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DOCS_DIR = os.path.join(DATA_DIR, "sample_docs")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_docs")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")

# Make sure these folders exist the first time the app runs. `exist_ok=True`
# means "don't raise an error if the folder is already there."
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CHUNKING PARAMETERS
# ---------------------------------------------------------------------------
# CHUNK_SIZE: the maximum number of characters in a single chunk of text.
# CHUNK_OVERLAP: how many characters from the end of one chunk are repeated
#   at the start of the next chunk.
#
# WHY 1000 / 150?
# A chunk needs to be:
#   - Small enough that it fits comfortably inside the LLM's context window
#     alongside several other chunks, the user's question, and the chat
#     history.
#   - Large enough that it still contains a *complete thought* (a chunk that
#     cuts a sentence in half loses meaning).
# 1000 characters (~150-200 English words) is a common sweet spot for
# paragraph-level enterprise documents (policies, manuals, reports).
#
# WHY OVERLAP?
# If a sentence that answers the user's question happens to sit right on the
# boundary between chunk N and chunk N+1, a hard cut with zero overlap could
# split it so badly that neither chunk contains the full idea. A 150-character
# overlap means the last ~1-2 sentences of a chunk reappear at the start of
# the next chunk, so boundary-spanning ideas survive intact in at least one
# chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# RETRIEVAL PARAMETERS
# ---------------------------------------------------------------------------
# TOP_K_RESULTS: how many of the most similar chunks we pull from ChromaDB
# for every question. Too low -> the LLM might miss the answer. Too high ->
# you flood the LLM's context with irrelevant text, increasing cost and the
# chance of the LLM getting confused ("lost in the middle" problem).
TOP_K_RESULTS = 4

# ---------------------------------------------------------------------------
# EMBEDDING + LLM MODEL NAMES
# ---------------------------------------------------------------------------
# We use Google's Gemini family via LangChain's `langchain-google-genai`
# integration package. You could swap these two lines to use OpenAI,
# HuggingFace, or Anthropic models instead — every other file only ever
# talks to LangChain's generic interfaces, never to Gemini directly, so a
# provider swap is a two-line change.
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.5-flash"

# Controls "creativity" of the LLM. 0 = fully deterministic/factual,
# 1 = very creative/random. For an enterprise fact-lookup tool we want low
# temperature so answers stay grounded in the retrieved text.
LLM_TEMPERATURE = 0.2

# Name of the collection (like a "table") inside ChromaDB where all vectors
# for this app are stored.
CHROMA_COLLECTION_NAME = "enterprise_knowledge_hub"

# Supported file extensions for upload
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".html", ".htm"]

# Read the Gemini API key from the environment (loaded from .env above).
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
