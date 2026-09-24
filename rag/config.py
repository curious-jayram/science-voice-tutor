"""Configuration for the NCERT File Search store."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RAGSettings:
    data_dir: Path = Path(os.getenv("NCERT_DATA_DIR", str(PROJECT_ROOT / "data")))
    index_dir: Path = Path(os.getenv("RAG_INDEX_DIR", str(PROJECT_ROOT / ".rag_index")))

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "file_search.json"


def load_file_search_store(settings: RAGSettings | None = None) -> str:
    """Return the File Search store that holds the NCERT chapters."""
    settings = settings or RAGSettings()
    configured = os.getenv("FILE_SEARCH_STORE", "").strip()
    if configured:
        return configured
    if not settings.manifest_path.exists():
        raise RuntimeError(
            "The NCERT File Search store has not been created. "
            "Run `python -m rag.indexer` first."
        )
    payload = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    store_name = str(payload.get("store_name") or "").strip()
    if not store_name:
        raise RuntimeError(
            "The NCERT File Search manifest has no store. Run `python -m rag.indexer`."
        )
    return store_name
