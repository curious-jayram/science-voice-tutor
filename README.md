# NCERT Science Voice Tutor

A FastAPI and Pipecat voice tutor for NCERT Science Classes 6–10. Sarvam provides speech
recognition and speech synthesis. Gemini 3.8 Flash answers from the textbooks through
Google File Search, which chunks, embeds, and retrieves passages inside one model call.

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `SARVAM_API_KEY` and `GOOGLE_API_KEY` in `.env`. Put one PDF per chapter under `data/`.
The filename must contain the class, chapter number, and chapter title:

```text
data/
  Class 6/Class 6 Chapter 1 The Wonderful World of Science.pdf
  Class 10/Class 10 Chapter 1 Chemical Reactions and Equations.pdf
```

The folder name must match the class in the filename.

## Build the File Search store

```powershell
python -m rag.indexer
```

The command uploads each PDF and lets Google File Search chunk and index it. It does not
embed or chunk the books locally. Progress is saved after every completed chapter in
`.rag_index/file_search.json`. Later runs skip files that have not changed.

On the Gemini free tier, a daily quota error stops the command at the chapter that failed.
Completed chapters stay in the store. Rerun the same command after the quota resets and it
continues with the remaining files. Use `--force` to upload every PDF again.

Do not start the voice tutor until the upload finishes. The tutor reads the store name
from the manifest. Set `FILE_SEARCH_STORE` only when you want to point at a store created
elsewhere.

## Run

```powershell
python main.py
```

Open `http://127.0.0.1:7860` to use the bundled WebRTC client. The client
creates a session at `POST /start`, then exchanges the WebRTC offer at
`/sessions/{sessionId}/api/offer`. FastAPI also exposes a health check at
`/health` and interactive API documentation at `/docs`. Set `HOST` or `PORT`
to override the default bind address.

For development with automatic reload:

```powershell
uvicorn main:app --reload --port 7860
```

Each spoken science question is answered in one Gemini request. File Search retrieves
the textbook passages while that request is running.

## Test

```powershell
pytest -q
```

Tests use a fake File Search client and do not make paid API calls.
