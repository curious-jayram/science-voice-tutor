"""Upload NCERT chapter PDFs to a Gemini File Search store.

Google chunks, embeds, and indexes each file. This command only sends the PDF
and the class, chapter number, and title taken from its filename. Completed
files are checkpointed so a free-tier daily quota stop can resume later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import errors

from rag.config import RAGSettings

CHAPTER_FILENAME = re.compile(
    r"^Class\s+(?P<class_level>\d+)\s+Chapter\s+(?P<chapter_number>\d+)\s+(?P<chapter_title>.+?)\s*$",
    re.IGNORECASE,
)
RETRY_DELAY = re.compile(
    r"retry in ([\d.]+)\s*(hours|hour|h|minutes|minute|mins|min|m|seconds|second|secs|sec|s)\b",
    re.IGNORECASE,
)


class DailyQuotaExhausted(Exception):
    """The Gemini free-tier daily quota is exhausted."""


@dataclass(frozen=True)
class ChapterFile:
    path: Path
    source: str
    class_level: int
    chapter_number: int
    chapter_title: str

    @property
    def display_name(self) -> str:
        return (
            f"Class {self.class_level} Chapter {self.chapter_number} "
            f"{self.chapter_title}"
        )


@dataclass
class IndexStats:
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    paused: bool = False
    stopped_before: str = ""
    duplicates: list[str] | None = None
    removed_copies: int = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as pdf:
        for block in iter(lambda: pdf.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_chapter_file(path: Path, data_dir: Path) -> ChapterFile:
    """Read class, chapter number, and title from a filename.

    Expected name: ``Class 10 Chapter 1 Chemical Reactions and Equations.pdf``.
    """
    try:
        relative = path.relative_to(data_dir)
    except ValueError as exc:
        raise ValueError(f"{path} is not inside {data_dir}") from exc

    match = CHAPTER_FILENAME.match(path.stem)
    if match is None:
        raise ValueError(
            f"Cannot read class, chapter, and title from {relative.as_posix()}. "
            "Name the file 'Class N Chapter M Title.pdf'."
        )

    class_level = int(match.group("class_level"))
    chapter_number = int(match.group("chapter_number"))
    if class_level not in range(6, 11):
        raise ValueError(f"{relative.as_posix()} must be Class 6, 7, 8, 9, or 10")
    if chapter_number < 1:
        raise ValueError(f"{relative.as_posix()} needs a positive chapter number")

    folder_match = re.fullmatch(r"Class\s+(\d+)", path.parent.name, re.IGNORECASE)
    if folder_match is None or int(folder_match.group(1)) != class_level:
        raise ValueError(
            f"{relative.as_posix()} is in '{path.parent.name}', "
            f"which does not match Class {class_level}"
        )

    return ChapterFile(
        path=path,
        source=relative.as_posix(),
        class_level=class_level,
        chapter_number=chapter_number,
        chapter_title=match.group("chapter_title").strip(),
    )


def document_is_active(state) -> bool:
    label = getattr(state, "name", None) or str(state or "")
    return label.endswith("ACTIVE")


def document_created(document) -> datetime:
    created = getattr(document, "create_time", None)
    if isinstance(created, datetime):
        return created if created.tzinfo else created.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def reconcile_chapter(
    chapter: ChapterFile,
    recorded_name: str,
    pending: dict | None,
    documents: list,
    *,
    replace: bool,
) -> tuple[list[str], str | None, bool]:
    """Decide which copies to delete and whether this chapter must be uploaded.

    A chapter keeps one finished document. The newest copy is removed when an
    earlier run was interrupted or when the same chapter was embedded twice.
    The chapter is uploaded only when no finished copy remains.
    """
    matches = sorted(
        (
            document
            for document in documents
            if getattr(document, "display_name", None) == chapter.display_name
        ),
        key=document_created,
    )
    interrupted = bool(pending) and pending.get("display_name") == chapter.display_name
    if not matches:
        return [], None, True

    if replace:
        return [document.name for document in matches], None, True

    if interrupted:
        newest = matches[-1]
        earlier = [
            document
            for document in matches[:-1]
            if document_is_active(getattr(document, "state", None))
        ]
        deletes = [newest.name]
        if earlier:
            keep = earlier[0]
            deletes.extend(
                document.name for document in matches[:-1] if document.name != keep.name
            )
            return deletes, keep.name, False
        deletes.extend(document.name for document in matches[:-1])
        return deletes, None, True

    active = [
        document
        for document in matches
        if document_is_active(getattr(document, "state", None))
    ]
    if not active:
        return [document.name for document in matches], None, True

    keep = next((document for document in active if document.name == recorded_name), active[0])
    deletes = [document.name for document in matches if document.name != keep.name]
    return deletes, keep.name, False


def discover_chapters(data_dir: Path) -> list[ChapterFile]:
    paths = sorted(data_dir.glob("Class */*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No NCERT PDFs found under {data_dir}")
    chapters = [parse_chapter_file(path, data_dir) for path in paths]
    chapters.sort(key=lambda chapter: (chapter.class_level, chapter.chapter_number, chapter.source))
    return chapters


def _error_text(exc: BaseException | dict | str) -> str:
    if isinstance(exc, dict):
        return json.dumps(exc)
    message = getattr(exc, "message", None)
    if message:
        return f"{exc} {message}"
    return str(exc)


def retry_delay_seconds(detail: str) -> float | None:
    match = RETRY_DELAY.search(detail)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("h"):
        return value * 3600
    if unit.startswith("m"):
        return value * 60
    return value


def is_daily_quota(detail: str, code: int | None = None) -> bool:
    lowered = detail.lower()
    if "perday" in lowered or "per day" in lowered or "per_day" in lowered:
        return True
    if "daily" in lowered and "quota" in lowered:
        return True
    delay = retry_delay_seconds(detail)
    quota_error = code == 429 or "resource_exhausted" in lowered or "quota" in lowered
    if quota_error and delay is not None and delay >= 30 * 60:
        return True
    if quota_error and delay is None and "quota" in lowered:
        return True
    return False


def is_short_rate_limit(detail: str, code: int | None = None) -> bool:
    if is_daily_quota(detail, code):
        return False
    delay = retry_delay_seconds(detail)
    quota_error = code == 429 or "resource_exhausted" in detail.lower()
    return quota_error and delay is not None and delay < 30 * 60


class NCERTIndexer:
    def __init__(
        self,
        settings: RAGSettings | None = None,
        client: genai.Client | None = None,
        poll_interval: float = 5.0,
    ):
        self.settings = settings or RAGSettings()
        self._client = client
        self.poll_interval = poll_interval

    def _api(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def _load_manifest(self) -> dict:
        try:
            return json.loads(self.settings.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 1, "store_name": "", "files": {}}

    def _save_manifest(
        self,
        store_name: str,
        files: dict[str, dict[str, str]],
        pending: dict | None = None,
    ) -> None:
        self.settings.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "store_name": store_name,
            "files": files,
            "pending": pending,
        }
        temporary = self.settings.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.settings.manifest_path)

    def _list_documents(self, store_name: str) -> list:
        client = self._api()
        return list(
            self._call(lambda: client.file_search_stores.documents.list(parent=store_name))
        )

    def _delete_document(self, name: str) -> None:
        client = self._api()
        self._call(
            lambda: client.file_search_stores.documents.delete(
                name=name, config={"force": True}
            )
        )

    def _ensure_store(self, manifest: dict) -> str:
        client = self._api()
        store_name = os.getenv("FILE_SEARCH_STORE", "").strip() or str(
            manifest.get("store_name") or ""
        ).strip()
        if store_name:
            self._call(lambda: client.file_search_stores.get(name=store_name))
            return store_name
        store = self._call(
            lambda: client.file_search_stores.create(
                config={"display_name": "NCERT Science Classes 6-10"}
            )
        )
        return store.name

    def _call(self, action):
        for attempt in range(6):
            try:
                return action()
            except errors.APIError as exc:
                detail = _error_text(exc)
                if is_daily_quota(detail, getattr(exc, "code", None)):
                    raise DailyQuotaExhausted(detail) from exc
                if is_short_rate_limit(detail, getattr(exc, "code", None)) and attempt < 5:
                    delay = retry_delay_seconds(detail) or 30.0
                    print(f"Gemini rate limit reached; retrying in {delay:.1f}s...", flush=True)
                    time.sleep(delay)
                    continue
                raise

    def _wait(self, operation):
        client = self._api()
        while not operation.done:
            if self.poll_interval:
                time.sleep(self.poll_interval)
            operation = self._call(lambda: client.operations.get(operation))
        if operation.error:
            detail = _error_text(operation.error)
            code = operation.error.get("code") if isinstance(operation.error, dict) else None
            if is_daily_quota(detail, code):
                raise DailyQuotaExhausted(detail)
            raise RuntimeError(f"File Search import failed: {detail}")
        return operation

    def _upload_chapter(self, store_name: str, chapter: ChapterFile):
        client = self._api()
        operation = self._call(
            lambda: client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_name,
                file=chapter.path,
                config={
                    "display_name": chapter.display_name,
                    "custom_metadata": [
                        {"key": "class_level", "numeric_value": chapter.class_level},
                        {
                            "key": "chapter_number",
                            "numeric_value": chapter.chapter_number,
                        },
                        {"key": "chapter_title", "string_value": chapter.chapter_title},
                        {"key": "source", "string_value": chapter.source},
                    ],
                },
            )
        )
        operation = self._wait(operation)
        if operation.response is not None and operation.response.document_name:
            return operation.response.document_name
        return ""

    def index_all(self, force: bool = False) -> IndexStats:
        chapters = discover_chapters(self.settings.data_dir)
        stats = IndexStats(discovered=len(chapters))
        manifest = self._load_manifest()
        files = {
            source: dict(record)
            for source, record in dict(manifest.get("files") or {}).items()
        }
        pending = manifest.get("pending") or None
        try:
            store_name = self._ensure_store(manifest)
            documents = self._list_documents(store_name)
        except DailyQuotaExhausted:
            stats.paused = True
            stats.stopped_before = (
                str(pending.get("source"))
                if pending and pending.get("source")
                else chapters[0].source
            )
            return stats

        titles: dict[str, int] = {}
        for document in documents:
            title = getattr(document, "display_name", None)
            if title:
                titles[title] = titles.get(title, 0) + 1
        stats.duplicates = sorted(title for title, count in titles.items() if count > 1)
        self._save_manifest(store_name, files, pending)

        ordered = list(chapters)
        if pending and pending.get("source"):
            ordered.sort(key=lambda chapter: chapter.source != pending["source"])

        for chapter in ordered:
            digest = file_sha256(chapter.path)
            previous = files.get(chapter.source) or {}
            replace = force or (
                bool(previous.get("sha256")) and previous.get("sha256") != digest
            )
            deletes, keep, needs_upload = reconcile_chapter(
                chapter,
                str(previous.get("document_name") or ""),
                pending if pending and pending.get("source") == chapter.source else None,
                documents,
                replace=replace,
            )
            if deletes:
                print(
                    f"Removing {len(deletes)} extra copy of {chapter.display_name}",
                    flush=True,
                )
            try:
                for name in deletes:
                    self._delete_document(name)
                    documents = [
                        document
                        for document in documents
                        if getattr(document, "name", None) != name
                    ]
                    stats.removed_copies += 1
            except DailyQuotaExhausted:
                stats.paused = True
                stats.stopped_before = chapter.source
                return stats

            if not needs_upload:
                if keep and previous.get("sha256") != digest:
                    files[chapter.source] = {"sha256": digest, "document_name": keep}
                    self._save_manifest(store_name, files, None)
                elif keep and previous.get("document_name") != keep:
                    files[chapter.source] = {"sha256": digest, "document_name": keep}
                    self._save_manifest(store_name, files, None)
                if pending and pending.get("source") == chapter.source:
                    pending = None
                    self._save_manifest(store_name, files, None)
                stats.unchanged += 1
                continue

            if not replace and previous.get("sha256") == digest and not (
                pending and pending.get("source") == chapter.source
            ):
                stats.unchanged += 1
                continue

            pending = {"source": chapter.source, "display_name": chapter.display_name}
            self._save_manifest(store_name, files, pending)
            print(f"Uploading {chapter.display_name}", flush=True)
            try:
                document_name = self._upload_chapter(store_name, chapter)
            except DailyQuotaExhausted:
                stats.paused = True
                stats.stopped_before = chapter.source
                return stats

            files[chapter.source] = {
                "sha256": digest,
                "document_name": document_name or keep or "",
            }
            pending = None
            self._save_manifest(store_name, files, None)
            stats.indexed += 1

        self._save_manifest(store_name, files, None)
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload every PDF again, including files already in the store",
    )
    args = parser.parse_args()
    stats = NCERTIndexer().index_all(force=args.force)
    if stats.duplicates:
        print("Duplicate documents: " + "; ".join(stats.duplicates))
    else:
        print("Duplicate documents: none")
    print(
        f"PDFs: {stats.discovered}; uploaded: {stats.indexed}; "
        f"unchanged: {stats.unchanged}; extra copies removed: {stats.removed_copies}"
    )
    if stats.paused:
        print(
            "Daily Gemini quota is exhausted. Completed chapters are saved. "
            "Rerun `python -m rag.indexer` after the quota resets to continue"
            + (f" from {stats.stopped_before}." if stats.stopped_before else ".")
        )


if __name__ == "__main__":
    main()
