"""Feedback store: save user feedback and model outputs for fine-tuning or prompt tuning."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class FeedbackEntry:
    """Single feedback record: prompt, response, user feedback, metadata."""

    id: Optional[str] = None
    session_id: Optional[str] = None
    prompt: str = ""
    response: str = ""
    feedback: Optional[str] = None  # thumbs_up | thumbs_down | or numeric score "1"-"5"
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "response": self.response,
            "feedback": self.feedback,
            "score": self.score,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeedbackEntry":
        return cls(
            id=d.get("id"),
            session_id=d.get("session_id"),
            prompt=d.get("prompt", ""),
            response=d.get("response", ""),
            feedback=d.get("feedback"),
            score=d.get("score"),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at"),
        )


class FeedbackStore:
    """File-based feedback store (JSONL). Append-only for fine-tuning export."""

    def __init__(self, path: str = "./data/feedback/feedback.jsonl"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        prompt: str,
        response: str,
        feedback: Optional[str] = None,
        score: Optional[float] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Append a feedback entry. Returns entry id (timestamp-based)."""
        from uuid import uuid4
        entry_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
        created = datetime.now(timezone.utc).strftime(UTC_FMT)
        entry = FeedbackEntry(
            id=entry_id,
            session_id=session_id,
            prompt=prompt,
            response=response,
            feedback=feedback,
            score=score,
            metadata=metadata or {},
            created_at=created,
        )
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry_id

    def list_entries(
        self,
        limit: int = 100,
        session_id: Optional[str] = None,
        has_feedback: Optional[bool] = None,
    ) -> list[FeedbackEntry]:
        """Read recent entries (last N lines). Optional filter by session_id or has_feedback."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        entries = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                e = FeedbackEntry.from_dict(d)
                if session_id is not None and e.session_id != session_id:
                    continue
                if has_feedback is not None:
                    if has_feedback and not e.feedback and e.score is None:
                        continue
                    if not has_feedback and (e.feedback or e.score is not None):
                        continue
                entries.append(e)
                if len(entries) >= limit:
                    break
            except Exception:
                continue
        return entries

    def export_for_finetuning(
        self,
        path: Optional[str] = None,
        format: str = "jsonl",
        only_with_feedback: bool = True,
    ) -> str:
        """Export entries to JSONL (messages format for fine-tuning). Returns path written."""
        out_path = Path(path or str(self._path.parent / "export_finetune.jsonl"))
        entries = self.list_entries(limit=100_000, has_feedback=only_with_feedback if only_with_feedback else None)
        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                rec = {
                    "messages": [
                        {"role": "user", "content": e.prompt},
                        {"role": "assistant", "content": e.response},
                    ],
                    "feedback": e.feedback,
                    "score": e.score,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(out_path)
