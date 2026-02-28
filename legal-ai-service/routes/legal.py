"""
FastAPI route handlers for NyayaDoot.

request  →  classify  →  retrieve (FAISS + Kanoon)  →  ONE LLM call  →  response
                 ↑ local                 ↑ no LLM              ↑ 256 tokens max
"""

import uuid
import re
import html
import time
from typing import List, Dict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.conversation import conversation_manager
from services.search import search_sections, parse_cases_from_llm
from services.llm import (
    is_legal_query,
    is_impact_query,
    is_follow_up,
    generate_legal_response,
    generate_impact_response,
    build_legal_context,
    build_case_context,
)

router = APIRouter(prefix="/Nyayadoot")


# ── Schemas ─────────────────────────────────────────────────────────────────

class ProcessQueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class ResponseModel(BaseModel):
    answer: str
    references: List[Dict]
    cases: List[Dict]
    is_follow_up: bool
    session_id: str


# ── Helpers ─────────────────────────────────────────────────────────────────

def _sanitize(query: str) -> str:
    return re.sub(r"\s+", " ", html.escape(query.strip()))


def _format_answer(raw: str, cases: List[Dict]) -> str:
    """Append clickable case-law URLs only if we fetched cases."""
    if not cases:
        return raw
    parts = [raw, "\n\n**Case References:**"]
    for c in cases[:3]:
        url = c.get("url", "")
        title = c.get("title", "Untitled")
        if url:
            parts.append(f"- [{title}]({url})")
        else:
            parts.append(f"- {title}")
    return "\n".join(parts)


# ── Main endpoint ───────────────────────────────────────────────────────────

@router.post("/process-query", response_model=ResponseModel)
async def process_legal_query(request: ProcessQueryRequest):
    try:
        t0 = time.perf_counter()

        # 1. Session handling
        sid = request.session_id
        if sid == "new_chat":
            sid = str(uuid.uuid4())

        query = _sanitize(request.query)
        if len(query) < 3:
            raise HTTPException(400, "Please ask a more detailed question.")

        has_history = conversation_manager.has_history(sid)
        history = conversation_manager.get_history_string(sid)

        # 2. Local classification (zero API cost)
        is_first = not has_history
        if is_first and not is_legal_query(query):
            return ResponseModel(
                answer="I'm NyayaDoot — a legal assistant. Please ask me a question related to Indian law.",
                references=[], cases=[], is_follow_up=False, session_id=sid,
            )

        followup = is_follow_up(query, has_history)
        impact = is_impact_query(query)

        # Detect what the user explicitly wants
        q_lower = query.lower()
        wants_sections = any(
            kw in q_lower
            for kw in ("law", "section", "act", "provision", "legal provision",
                       "which law", "what law", "relevant law")
        )
        wants_cases = any(
            kw in q_lower
            for kw in ("case", "precedent", "judgment", "ruling", "court case", "case law")
        )

        # 3. Retrieval (no LLM calls)
        #    Always do FAISS search (cheap) so the LLM has legal context
        sections = search_sections(query, history if followup else "")

        # 4. Single LLM call (≤ 200 tokens)
        #    Cases are now suggested BY the LLM (no scraping) when wants_cases=True
        if impact:
            raw_answer = generate_impact_response(query, history, sections)
        else:
            raw_answer = generate_legal_response(
                query, history, sections, [],
                show_sections=wants_sections,
                show_cases=wants_cases,
            )

        # 5. Extract case names from LLM response and attach Kanoon search links
        cases: List[Dict] = []
        if wants_cases:
            cases = parse_cases_from_llm(raw_answer)

        # 6. Format and persist
        formatted = _format_answer(raw_answer, cases)
        conversation_manager.save_turn(sid, query, formatted)

        elapsed = time.perf_counter() - t0
        print(f"[route] total request time: {elapsed:.2f}s")

        return ResponseModel(
            answer=formatted,
            references=sections if wants_sections else [],
            cases=cases,
            is_follow_up=followup,
            session_id=sid,
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[route] error: {exc}")
        raise HTTPException(500, detail=str(exc))


# ── Reset session ───────────────────────────────────────────────────────────

@router.post("/reset-session")
async def reset_session(session_id: str = Query(...)):
    conversation_manager.reset(session_id)
    return {"status": "reset"}
