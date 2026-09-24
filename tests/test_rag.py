from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors

from rag.config import RAGSettings, load_file_search_store
from rag.indexer import NCERTIndexer, parse_chapter_file


def make_settings(tmp_path: Path) -> RAGSettings:
    return RAGSettings(data_dir=tmp_path / "data", index_dir=tmp_path / "index")


def write_chapter(data_dir: Path, class_level: int, chapter_number: int, title: str, body: bytes) -> Path:
    directory = data_dir / f"Class {class_level}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"Class {class_level} Chapter {chapter_number} {title}.pdf"
    path.write_bytes(body)
    return path


class FakeClient:
    def __init__(self, fail_on: int | None = None):
        self.fail_on = fail_on
        self.creates = 0
        self.gets: list[str] = []
        self.configs: list[dict] = []
        self.remote: list = []
        self.file_search_stores = self.Stores(self)
        self.operations = self.Operations()

    class Operations:
        def get(self, operation):
            return operation

    class Stores:
        def __init__(self, owner: FakeClient):
            self.owner = owner
            self.documents = self.Documents(owner)

        class Documents:
            def __init__(self, owner: FakeClient):
                self.owner = owner
                self.deleted: list[str] = []

            def delete(self, *, name: str, config=None) -> None:
                assert config == {"force": True}
                self.deleted.append(name)
                self.owner.remote = [
                    document
                    for document in self.owner.remote
                    if getattr(document, "name", None) != name
                ]

            def list(self, *, parent: str):
                return list(self.owner.remote)

        def create(self, *, config=None):
            self.owner.creates += 1
            return SimpleNamespace(name="fileSearchStores/ncert")

        def get(self, *, name: str):
            self.owner.gets.append(name)
            return SimpleNamespace(name=name)

        def upload_to_file_search_store(self, *, file_search_store_name, file, config=None):
            self.owner.configs.append(config)
            index = len(self.owner.configs)
            if self.owner.fail_on == index:
                raise errors.ClientError(
                    429,
                    {
                        "error": {
                            "code": 429,
                            "message": "Quota exceeded for embed_content_free_tier_requests PerDay",
                            "status": "RESOURCE_EXHAUSTED",
                        }
                    },
                )
            return SimpleNamespace(
                done=True,
                error=None,
                name=f"operations/{index}",
                response=SimpleNamespace(
                    document_name=f"{file_search_store_name}/documents/{index}"
                ),
            )


def test_parse_chapter_file_uses_the_filename(tmp_path):
    data_dir = tmp_path / "data"
    path = write_chapter(
        data_dir,
        10,
        9,
        "Light – Reflection and Refraction",
        b"pdf",
    )
    chapter = parse_chapter_file(path, data_dir)
    assert chapter.class_level == 10
    assert chapter.chapter_number == 9
    assert chapter.chapter_title == "Light – Reflection and Refraction"
    assert chapter.source == "Class 10/Class 10 Chapter 9 Light – Reflection and Refraction.pdf"

    mismatched = data_dir / "Class 6" / path.name
    mismatched.parent.mkdir(parents=True, exist_ok=True)
    mismatched.write_bytes(b"pdf")
    with pytest.raises(ValueError, match="does not match"):
        parse_chapter_file(mismatched, data_dir)


def test_quota_stop_resumes_without_reuploading(tmp_path, monkeypatch):
    monkeypatch.delenv("FILE_SEARCH_STORE", raising=False)
    settings = make_settings(tmp_path)
    write_chapter(settings.data_dir, 6, 1, "The Wonderful World of Science", b"one")
    write_chapter(settings.data_dir, 6, 2, "Diversity in the Living World", b"two")

    first_client = FakeClient(fail_on=2)
    first = NCERTIndexer(settings, first_client, poll_interval=0).index_all()
    assert first.indexed == 1
    assert first.paused is True
    assert first.stopped_before.endswith("Chapter 2 Diversity in the Living World.pdf")
    assert "chunking_config" not in first_client.configs[0]
    assert first_client.configs[0]["custom_metadata"][0] == {
        "key": "class_level",
        "numeric_value": 6,
    }

    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["store_name"] == "fileSearchStores/ncert"
    assert len(manifest["files"]) == 1

    second_client = FakeClient()
    second = NCERTIndexer(settings, second_client, poll_interval=0).index_all()
    assert second.indexed == 1
    assert second.unchanged == 1
    assert second.paused is False
    assert second_client.creates == 0
    assert second_client.gets == ["fileSearchStores/ncert"]
    assert len(json.loads(settings.manifest_path.read_text(encoding="utf-8"))["files"]) == 2


def test_duplicate_active_copy_is_removed_without_uploading_again(tmp_path, monkeypatch):
    monkeypatch.delenv("FILE_SEARCH_STORE", raising=False)
    settings = make_settings(tmp_path)
    write_chapter(settings.data_dir, 10, 2, "Acids, Bases and Salts", b"acids")
    source = "Class 10/Class 10 Chapter 2 Acids, Bases and Salts.pdf"
    display_name = "Class 10 Chapter 2 Acids, Bases and Salts"
    settings.index_dir.mkdir()
    settings.manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "store_name": "fileSearchStores/ncert",
                "files": {
                    source: {
                        "sha256": hashlib.sha256(b"acids").hexdigest(),
                        "document_name": "fileSearchStores/ncert/documents/newer",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    client = FakeClient()
    client.remote = [
        SimpleNamespace(
            name="fileSearchStores/ncert/documents/older",
            display_name=display_name,
            state="STATE_ACTIVE",
            create_time=datetime(2026, 9, 22, 12, 24, tzinfo=timezone.utc),
        ),
        SimpleNamespace(
            name="fileSearchStores/ncert/documents/newer",
            display_name=display_name,
            state="STATE_ACTIVE",
            create_time=datetime(2026, 9, 22, 13, 27, tzinfo=timezone.utc),
        ),
    ]

    stats = NCERTIndexer(settings, client, poll_interval=0).index_all()

    assert stats.duplicates == [display_name]
    assert stats.indexed == 0
    assert stats.unchanged == 1
    assert client.file_search_stores.documents.deleted == [
        "fileSearchStores/ncert/documents/older"
    ]
    assert client.configs == []


def test_interrupted_upload_is_deleted_and_sent_once(tmp_path, monkeypatch):
    monkeypatch.delenv("FILE_SEARCH_STORE", raising=False)
    settings = make_settings(tmp_path)
    write_chapter(settings.data_dir, 6, 2, "Diversity in the Living World", b"two")
    display_name = "Class 6 Chapter 2 Diversity in the Living World"
    settings.index_dir.mkdir()
    settings.manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "store_name": "fileSearchStores/ncert",
                "files": {},
                "pending": {
                    "source": "Class 6/Class 6 Chapter 2 Diversity in the Living World.pdf",
                    "display_name": display_name,
                },
            }
        ),
        encoding="utf-8",
    )
    client = FakeClient()
    client.remote = [
        SimpleNamespace(
            name="fileSearchStores/ncert/documents/partial",
            display_name=display_name,
            state="STATE_PENDING",
            create_time=datetime(2026, 9, 23, 4, 0, tzinfo=timezone.utc),
        )
    ]

    stats = NCERTIndexer(settings, client, poll_interval=0).index_all()

    assert client.file_search_stores.documents.deleted == [
        "fileSearchStores/ncert/documents/partial"
    ]
    assert stats.indexed == 1
    assert len(client.configs) == 1
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["pending"] is None
    assert manifest["files"]


def test_missing_store_explains_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.delenv("FILE_SEARCH_STORE", raising=False)
    settings = make_settings(tmp_path)
    with pytest.raises(RuntimeError, match="python -m rag.indexer"):
        load_file_search_store(settings)
