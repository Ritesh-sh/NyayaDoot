"""
LangChain-powered LLM layer.

Key design:
 • ONE Gemini call per user query (≤250 token budget).
 • Context (FAISS sections, conversation history) is injected
   into a single prompt template so the model sees everything at once.
 • Legal-query classification is done locally (regex/keyword) — zero API cost.
 • Separate prompt templates for each response shape so small models
   don't need to parse conditional logic.
"""

from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    MAX_OUTPUT_TOKENS,
    LLM_TEMPERATURE,
    MAX_HISTORY_CHARS,
)

# ── Singleton LLM instance ─────────────────────────────────────────────────
_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    temperature=LLM_TEMPERATURE,
)

# ════════════════════════════════════════════════════════════════════════════
# Prompt templates — one per response "shape" (no conditional logic inside)
# ════════════════════════════════════════════════════════════════════════════

_BASE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "legal_context", "query"],
    template="""You are NyayaDoot, an Indian legal assistant. Answer briefly.

{chat_history}

Legal context:
{legal_context}

User: {query}

Give a direct answer in 2-3 sentences and up to 3 action steps. Nothing else.""",
)

_SECTIONS_PROMPT = PromptTemplate(
    input_variables=["chat_history", "legal_context", "query"],
    template="""You are NyayaDoot, an Indian legal assistant.

{chat_history}

Legal context:
{legal_context}

User: {query}

Give a 1-2 sentence answer. Then list exactly 3 most relevant legal sections as:
- Section X of Act: one-line explanation
No extra text.""",
)

_CASES_PROMPT = PromptTemplate(
    input_variables=["chat_history", "legal_context", "query"],
    template="""You are NyayaDoot, an Indian legal assistant.

{chat_history}

Legal context:
{legal_context}

User: {query}

Give a 1-2 sentence answer. Then list exactly 3 real Indian court cases as:
- Case Name v. Opponent (Year) - Court - relevance
No extra text.""",
)

_FULL_PROMPT = PromptTemplate(
    input_variables=["chat_history", "legal_context", "query"],
    template="""You are NyayaDoot, an Indian legal assistant.

{chat_history}

Legal context:
{legal_context}

User: {query}

Give a 1-2 sentence answer. Then list exactly 3 relevant sections as:
- Section X of Act: explanation
Then list exactly 3 real Indian court cases as:
- Case Name v. Opponent (Year) - Court - relevance
No extra text.""",
)

_IMPACT_PROMPT = PromptTemplate(
    input_variables=["chat_history", "legal_context", "query"],
    template="""You are NyayaDoot, an Indian legal assistant.
Analyze the impact of this legal situation briefly.

{chat_history}

Legal context:
{legal_context}

User: {query}

Respond with:
- Immediate impact (2 bullet points)
- Risk level (Low/Medium/High)
- 2-3 mitigation steps""",
)

# Pre-built chains (reused across requests) — LCEL pipe syntax
_parser = StrOutputParser()
_base_chain = _BASE_PROMPT | _llm | _parser
_sections_chain = _SECTIONS_PROMPT | _llm | _parser
_cases_chain = _CASES_PROMPT | _llm | _parser
_full_chain = _FULL_PROMPT | _llm | _parser
_impact_chain = _IMPACT_PROMPT | _llm | _parser


# ════════════════════════════════════════════════════════════════════════════
# Public helpers
# ════════════════════════════════════════════════════════════════════════════

def _trim(text: str, limit: int) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def build_legal_context(sections: List[Dict]) -> str:
    if not sections:
        return "No specific provisions retrieved."
    lines = []
    for s in sections:
        act = s["act"].replace(".pdf", "").replace("-", " ").title()
        lines.append(f"• {act}, Section {s['section_number']}: {s['summary']}")
    return "\n".join(lines)


def build_case_context(cases: List[Dict]) -> str:
    if not cases:
        return ""
    lines = []
    for c in cases:
        lines.append(f"• {c['title']} — {c.get('snippet', '')[:120]}")
    return "\n".join(lines)


def generate_legal_response(
    query: str,
    history: str,
    sections: List[Dict],
    cases: List[Dict],
    show_sections: bool = False,
    show_cases: bool = False,
) -> str:
    """Single LLM call — picks the right prompt based on what user asked for."""
    import time
    t0 = time.perf_counter()
    try:
        params = {
            "chat_history": _trim(history, MAX_HISTORY_CHARS) if history else "",
            "legal_context": build_legal_context(sections) if sections else "",
            "query": query,
        }
        # Pick the right chain — no conditional flags inside the prompt
        if show_sections and show_cases:
            chain = _full_chain
        elif show_sections:
            chain = _sections_chain
        elif show_cases:
            chain = _cases_chain
        else:
            chain = _base_chain

        result = chain.invoke(params)
        elapsed = time.perf_counter() - t0
        print(f"[llm] legal response in {elapsed:.2f}s")
        return result.strip()
    except Exception as exc:
        print(f"[llm] generation error: {exc}")
        return (
            "I'm having trouble generating a response right now. "
            "Please consult a qualified attorney for personalised advice."
        )


def generate_impact_response(
    query: str,
    history: str,
    sections: List[Dict],
) -> str:
    """Single LLM call for an impact / consequence query."""
    import time
    t0 = time.perf_counter()
    try:
        result = _impact_chain.invoke({
            "chat_history": _trim(history, MAX_HISTORY_CHARS) if history else "",
            "legal_context": build_legal_context(sections) if sections else "",
            "query": query,
        })
        elapsed = time.perf_counter() - t0
        print(f"[llm] impact response in {elapsed:.2f}s")
        return result.strip()
    except Exception as exc:
        print(f"[llm] impact error: {exc}")
        return (
            "Unable to generate an impact analysis at this time. "
            "Please consult a legal professional."
        )


# ════════════════════════════════════════════════════════════════════════════
# Local (zero-cost) query classification
# ════════════════════════════════════════════════════════════════════════════

_LEGAL_KEYWORDS = frozenset(
    "law legal right rights duty obligation liability contract agreement court "
    "judge judgment case precedent statute act section clause provision regulation "
    "rule criminal civil lawsuit sue attorney lawyer legislation penalty punishment "
    "bail arrest fir police complaint ipc crpc bns bnss constitution property "
    "divorce custody maintenance cheque dishonour defamation fraud theft murder "
    "dowry harassment eviction tenant landlord consumer complaint insurance claim "
    "accident compensation negligence damages injunction writ habeas corpus pil "
    "arbitration mediation settlement appeal revision".split()
)

_NON_LEGAL_PATTERNS = [
    "recipe", "weather", "joke", "song", "movie", "game", "sport",
    "cook", "play", "sing", "dance", "travel", "hotel", "restaurant",
    "hello", "hi ", "hey ", "good morning", "good night", "how are you",
]


def is_legal_query(query: str) -> bool:
    """Fast local classifier — no API call needed."""
    q = query.lower()
    # reject obvious non-legal
    if any(p in q for p in _NON_LEGAL_PATTERNS):
        return False
    words = set(q.split())
    # if any legal keyword present → legal
    if words & _LEGAL_KEYWORDS:
        return True
    # queries mentioning "what / how / can / is" + longer than 5 words → assume legal
    if len(words) >= 5 and any(w in words for w in ("what", "how", "can", "is", "when", "where")):
        return True
    # short ambiguous → default legal (better UX)
    return True


_IMPACT_KEYWORDS = frozenset(
    "impact effect consequence result outcome implication penalty punishment "
    "fine damage loss risk what-will-happen worst-case scenario".split()
)

_IMPACT_PHRASES = [
    "what will happen", "what happens if", "how will this affect",
    "what are the consequences", "how serious", "what are the risks",
    "what could happen", "worst case",
]


def is_impact_query(query: str) -> bool:
    q = query.lower()
    if any(p in q for p in _IMPACT_PHRASES):
        return True
    words = set(q.split())
    return bool(words & _IMPACT_KEYWORDS)


def is_follow_up(query: str, has_history: bool) -> bool:
    """Decide if query is a follow-up (local, no API)."""
    if not has_history:
        return False
    q = query.lower()
    # short queries with history are almost always follow-ups
    if 2 <= len(q.split()) <= 7:
        return True
    follow_indicators = [
        "follow up", "previous", "explain more", "tell me more",
        "about that", "about this", "elaborate", "continue",
        "what about", "how about", "also", "and then",
        "can you further", "more info", "clarify",
        "related to this", "related to that", "regarding this",
        "regarding that", "in this case", "for this", "for that",
        "give me", "show me", "tell me", "provide me",
        "same", "this situation", "that situation",
        "above", "mentioned", "earlier",
    ]
    return any(ind in q for ind in follow_indicators)
