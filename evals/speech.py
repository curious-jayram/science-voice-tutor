"""Sarvam HTTP voice for the student in audio evals.

The harness caches each spoken turn and calls ``run_tts`` directly, so this
has to be the HTTP service. The WebSocket service yields nothing on that path.
``kavya`` is the student. The tutor keeps ``shubh``, so the two speakers in a
recording are not the same voice.
"""

import os

import aiohttp
from dotenv import load_dotenv

from pipecat.services.sarvam.tts import SarvamHttpTTSService
from pipecat.transcriptions.language import Language

load_dotenv()


def sarvam(config: dict) -> SarvamHttpTTSService:
    """Build the Indian-English student voice from a ``user.speech`` block."""
    language = config.get("language") or os.getenv("SARVAM_LANGUAGE", "en-IN")
    voice = config.get("voice") or "kavya"
    return SarvamHttpTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        aiohttp_session=aiohttp.ClientSession(),
        settings=SarvamHttpTTSService.Settings(
            model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
            voice=voice,
            language=Language(language),
            pace=1.0,
        ),
    )
