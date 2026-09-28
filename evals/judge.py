"""Gemini factory for the eval judge and the simulated student."""

import os
from pathlib import Path

from dotenv import load_dotenv

from pipecat.services.google.llm import GoogleLLMService

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def gemini(config: dict) -> GoogleLLMService:
    """Build a Gemini service from a scenario ``factory:`` block.

    The harness passes the YAML mapping. ``model`` overrides ``GEMINI_LLM_MODEL``.
    Thinking stays on the lowest Gemini 3.8 Flash level so verdicts stay short.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
    model = config.get("model") or os.getenv("GEMINI_LLM_MODEL", "gemini-3.8-flash")
    return GoogleLLMService(
        api_key=api_key,
        settings=GoogleLLMService.Settings(
            model=model,
            temperature=0,
            thinking=GoogleLLMService.ThinkingConfig(
                thinking_level="low",
                include_thoughts=False,
            ),
            extra={"automatic_function_calling": {"disable": True}},
        ),
    )
