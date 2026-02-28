"""
One-time loading of heavy assets: SentenceTransformer, FAISS index, section data.
Exposes module-level singletons so every other module just imports them.
"""

import pickle
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_PATH, FAISS_INDEX_PATH, SECTION_DATA_PATH


def _load_all():
    print("Loading embedding model …")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    print("Loading FAISS index …")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    print("Loading section data …")
    with open(SECTION_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    print("All models loaded.")
    return embedding_model, faiss_index, data["section_data"], data["all_acts"]


embedding_model, faiss_index, section_data, all_acts = _load_all()
