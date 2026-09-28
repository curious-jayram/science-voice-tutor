# NCERT Science Voice Tutor

A FastAPI and Pipecat voice tutor for NCERT Science Classes 6–10. Sarvam provides speech
recognition and speech synthesis. Gemini 3.8 Flash answers from the textbooks through
Google File Search, which chunks, embeds, and retrieves passages inside one model call.

## Setup

Use Python 3.11 or newer.

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`requirements.txt` skips the Daily extra on Windows, because that library has no Windows wheel. The local client at `http://127.0.0.1:7860` uses WebRTC and does not need it. Linux and macOS installs include Daily, which `GET /call` and the Cloud image use.

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

Do not start the voice tutor until the upload finishes. A local tutor reads the store
name from the manifest when `FILE_SEARCH_STORE` is empty.

Pipecat Cloud does not include `.rag_index`. Copy `store_name` from
`.rag_index/file_search.json` into `FILE_SEARCH_STORE` in `.env` before the
secrets command below. Cloud reads only that variable.

## Run

```powershell
python main.py
```

Open `http://127.0.0.1:7860` for the local WebRTC client on this machine. The
client creates a session at `POST /start`, then exchanges the WebRTC offer at
`/sessions/{sessionId}/api/offer`.

The shareable voice path runs on Pipecat Cloud. Cloud creates a Daily room
for each session and starts `bot()` in `bot.py`. Deploy with the Pipecat CLI:

```powershell
pipecat cloud auth login
pipecat cloud secrets set ncert-science-tutor-secrets --file .env --skip
pipecat cloud deploy
pipecat cloud agent start ncert-science-tutor --use-daily
```

Open the Daily URL that start command prints, or use the agent Sandbox in the
Pipecat Cloud dashboard. Keep the camera off so the session stays a 1:1 voice
call.

FastAPI also exposes a health check at `/health` and interactive API
documentation at `/docs`. Set `HOST` or `PORT` to override the default bind
address.

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

## Evals

Behavioral evals drive the real tutor. Text mode still calls Gemini and File Search, and it skips Sarvam. Run them from the repo root after `python -m rag.indexer`.

One scenario against a bot you leave running:

```powershell
python -m evals.serve --port 7861
python -m pipecat.evals run scenarios/scripted scenarios/simulated --bot-url ws://localhost:7861 -v --logs-dir eval-runs
```

The full suite starts a fresh tutor per scenario. Text scenarios skip Sarvam. Audio scenarios speak the student with Sarvam `bulbul:v3` voice `kavya` (Indian English), run Sarvam speech recognition and the tutor voice `shubh`, and judge a Moonshine transcript of what was spoken.

```powershell
python -m pipecat.evals suite evals/manifest.yaml
```

Audio mode needs the Pipecat CLI and Moonshine, which are not part of the Cloud image. Install them into the same environment as the tutor so the Pipecat extras combine:

```bash
pip install -r requirements.txt -r requirements-evals.txt
```

Run only the speech scenarios with:

```powershell
python -m pipecat.evals suite evals/manifest.yaml -s audio
```
