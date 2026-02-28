"""
All retrieval logic lives here:
  1. FAISS semantic search over local legal sections
  2. Kanoon search-link generator (no scraping — Cloudflare blocks it)

No LLM calls are made in this module — summaries are generated locally
from the section text itself (zero API cost, instant).
"""

from typing import List, Dict
import re
import urllib.parse
import numpy as np

from config import FAISS_TOP_K
from services.model_loader import embedding_model, faiss_index, section_data


# ════════════════════════════════════════════════════════════════════════════
# 1.  FAISS section search
# ════════════════════════════════════════════════════════════════════════════

def _build_local_summary(text: str, act: str) -> str:
    """Generate a short summary from the section text itself — no LLM needed."""
    act_clean = act.replace(".pdf", "").replace("-", " ").title()
    text_lower = text.lower()

    indicators = ["shall", "may", "means", "includes", "provided that"]
    for kw in indicators:
        if kw in text_lower:
            phrase = text_lower.split(kw, 1)[1].split(".")[0].strip()[:80]
            if len(phrase) < 5:
                continue  # skip empty / too-short extractions
            prefix = {
                "shall": "Mandates that",
                "may": "Allows for",
                "means": "Defines",
                "includes": "Defines",
                "provided that": "Subject to condition that",
            }.get(kw, "States that")
            return f"{prefix} {phrase}"

    important = ["right", "duty", "obligation", "liability", "penalty",
                  "punishment", "compensation", "damages", "relief", "remedy"]
    for term in important:
        if term in text_lower:
            return f"Addresses {term} under {act_clean}"

    return f"Legal provision from {act_clean}"


def _extract_topic_from_history(history: str) -> str:
    """Pull the core legal topic from conversation history for better FAISS search.
    
    Instead of dumping the whole history (which drowns the signal), extract
    only the user's original question keywords.
    """
    if not history:
        return ""
    # grab only the User: lines and extract keywords
    user_parts = []
    for line in history.split("\n"):
        if line.startswith("User:"):
            user_parts.append(line.replace("User:", "").strip())
    topic_text = " ".join(user_parts)
    # extract keywords (remove stopwords)
    words = topic_text.lower().split()
    filtered = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    return " ".join(filtered[:8])  # max 8 keywords


def search_sections(query: str, history: str = "") -> List[Dict]:
    """Return top-K legal sections using FAISS cosine similarity.
    
    Summaries are computed locally from the section text (no API call).
    For follow-ups, we extract the topic from history instead of dumping raw text.
    """
    try:
        if history:
            topic = _extract_topic_from_history(history)
            search_text = f"{topic} {query}".strip()
        else:
            search_text = query
        print(f"[search] FAISS query: {search_text[:100]}")
        vec = embedding_model.encode([search_text])
        D, I = faiss_index.search(np.array(vec, dtype="float32"), FAISS_TOP_K)

        results = []
        for rank, idx in enumerate(I[0]):
            sec = section_data[idx]
            results.append({
                "act": sec["act"],
                "section_number": sec["section_number"],
                "full_text": sec["full_text"][:300],
                "summary": _build_local_summary(sec["full_text"], sec["act"]),
                "score": float(D[0][rank]),
            })
        return results
    except Exception as exc:
        print(f"[search] FAISS error: {exc}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# 2.  Kanoon search-link generator (replaces broken scraping)
# ════════════════════════════════════════════════════════════════════════════

def generate_case_link(case_name: str) -> str:
    """Build a Google Scholar search URL for a case name (reliable, unlike Indian Kanoon which often 522s)."""
    return f"https://scholar.google.com/scholar?q={urllib.parse.quote_plus(case_name)}"


def parse_cases_from_llm(llm_text: str) -> List[Dict]:
    """Extract case names from the LLM response and attach Kanoon search links.
    
    Works with formats like:
      - Case Name (Year) - Court - Relevance
      - **Case Name** (Year)
      - Case Name, Year
    """
    cases = []
    seen = set()
    # match lines that look like case citations
    patterns = [
        # "Name v. Name" or "Name vs Name" or "State of X v. Y"
        r'[\*\-\d\.\)]*\s*\*{0,2}([A-Z][\w\s\.]+(?:\bv[s]?\.?\s+)[A-Z][\w\s\.]+?)(?:\*{0,2})\s*[\(\-]',
        # fallback: line starting with - or * followed by capitalized text before ( or -
        r'[\-\*\u2022]\s*\*{0,2}([A-Z][\w\s\.]{10,60}?)(?:\*{0,2})\s*[\(\-]',
    ]
    _COURT_NOISE = {"supreme court", "high court", "hc", "sc", "district court",
                     "sessions court", "tribunal", "bench", "division bench",
                     "supreme court of india", "rajasthan high court",
                     "delhi high court", "bombay high court", "calcutta high court",
                     "madras high court", "karnataka high court", "allahabad high court"}
    # Only parse from the main body, strip the appended "Case References:" block
    body = llm_text.split("**Case References:**")[0] if "**Case References:**" in llm_text else llm_text
    # Process line-by-line so regex never spans across entries
    for line in body.splitlines():
        if len(cases) >= 3:
            break
        line = line.strip()
        if not line:
            continue
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                name = m.group(1).strip().rstrip("*- ")
                name_lower = name.lower()
                if (name and name not in seen and len(name) > 12
                        and name_lower not in _COURT_NOISE
                        and not all(w.lower() in _COURT_NOISE for w in name.split())):
                    seen.add(name)
                    cases.append({
                        "title": name,
                        "url": generate_case_link(name),
                        "snippet": "Search for full judgment on Google Scholar",
                    })
                    break  # one case per line
    return cases[:3]


# ─── lightweight keyword extraction (no LLM) ──────────────────────────────

_STOPWORDS = frozenset(
    "the a an and or but is are was were in on at to for with by about as of "
    "can could would should will shall may might must have has had do does did "
    "i me my you your he him his she her it its we us our they them their this "
    "that these those be been being if because until while from up down out over "
    "under again then once here there when where why how all any both each few "
    "more most other some no nor not only so than too very just now".split()
)


def _extract_keywords(history: str, query: str, max_kw: int = 6) -> str:
    from collections import Counter
    words = (history + " " + query).lower().split()
    filtered = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    common = [w for w, _ in Counter(filtered).most_common(max_kw)]
    return " ".join(common)
