"""
Session / conversation memory backed by LangChain InMemoryChatMessageHistory.
Each session_id maps to its own history object so multiple users stay isolated.
"""

from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from config import MAX_CONVERSATION_TURNS


class ConversationManager:
    """Thread-safe-ish session store (fine for single-process uvicorn)."""

    def __init__(self):
        self._sessions: Dict[str, InMemoryChatMessageHistory] = {}

    # ── internal ────────────────────────────────────────────────────────
    def _get_store(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
        return self._sessions[session_id]

    # ── public API ──────────────────────────────────────────────────────
    def get_history_string(self, session_id: str) -> str:
        """Return the last N turns as a plain string."""
        store = self._get_store(session_id)
        messages = store.messages[-(MAX_CONVERSATION_TURNS * 2):]
        parts = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content
            if len(content) > 400:
                content = content[:400] + "..."
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def save_turn(self, session_id: str, user_msg: str, ai_msg: str):
        store = self._get_store(session_id)
        store.add_message(HumanMessage(content=user_msg))
        store.add_message(AIMessage(content=ai_msg))

    def reset(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def has_history(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        return len(self._sessions[session_id].messages) > 0


# Module-level singleton
conversation_manager = ConversationManager()
