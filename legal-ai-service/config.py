"""
Centralized configuration for NyayaDoot Legal AI Service.
All environment variables, constants, and tunable parameters live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Model paths ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_PATH = "models/legal_embedding_model"
FAISS_INDEX_PATH = "models/legal_index.faiss"
SECTION_DATA_PATH = "models/legal_sections.pkl"

# ── LLM settings (single-call budget) ──────────────────────────────────────
GEMINI_MODEL_NAME = "gemma-3-4b-it"
MAX_OUTPUT_TOKENS = 300          # hard cap per query (enough for 3 items + answer)
LLM_TEMPERATURE = 0.4

# ── Search tuning ───────────────────────────────────────────────────────────
FAISS_TOP_K = 3                  # sections to retrieve
MAX_CONVERSATION_TURNS = 3       # history turns kept in memory
MAX_HISTORY_CHARS = 400          # char limit for history in prompts

# ── Server ──────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── CORS ────────────────────────────────────────────────────────────────────
CORS_ORIGINS = ["*"]
