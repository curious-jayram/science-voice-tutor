"""NCERT science voice tutor with Sarvam speech and Gemini File Search."""

import os
import re
from collections import deque

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    LLMTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from rag.config import load_file_search_store

load_dotenv()


class SpeechAudioGate(FrameProcessor):
    """Send microphone audio downstream only during Silero-detected speech."""

    def __init__(self, pre_roll_secs: float = 0.3):
        super().__init__()
        self._pre_roll_secs = pre_roll_secs
        self._pre_roll_duration = 0.0
        self._pre_roll: deque[tuple[InputAudioRawFrame, FrameDirection, float]] = deque()
        self._speech_active = False

    @staticmethod
    def _duration(frame: InputAudioRawFrame) -> float:
        bytes_per_sample = 2
        return len(frame.audio) / (
            frame.sample_rate * frame.num_channels * bytes_per_sample
        )

    def _buffer_audio(
        self, frame: InputAudioRawFrame, direction: FrameDirection
    ) -> None:
        duration = self._duration(frame)
        self._pre_roll.append((frame, direction, duration))
        self._pre_roll_duration += duration

        while self._pre_roll and self._pre_roll_duration > self._pre_roll_secs:
            _, _, removed_duration = self._pre_roll.popleft()
            self._pre_roll_duration -= removed_duration

    async def _flush_pre_roll(self) -> None:
        while self._pre_roll:
            frame, direction, _ = self._pre_roll.popleft()
            await self.push_frame(frame, direction)
        self._pre_roll_duration = 0.0

    async def process_frame(
        self, frame: Frame, direction: FrameDirection
    ) -> None:
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speech_active = True
            await self.push_frame(frame, direction)
            await self._flush_pre_roll()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.push_frame(frame, direction)
            self._speech_active = False
        elif isinstance(frame, InputAudioRawFrame):
            if self._speech_active:
                await self.push_frame(frame, direction)
            else:
                self._buffer_audio(frame, direction)
        else:
            await self.push_frame(frame, direction)


# Gemini writes File Search citations into the answer as
# [PerQueryResult(index='1.2')]. Those markers are for a UI, and TTS reads
# them aloud if they reach Sarvam.
_COMPLETE_CITATION = re.compile(
    r"[ \t]*\[(?:[ \t]*PerQueryResult\([ \t]*index[ \t]*=[ \t]*"
    r"(?:'[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?)[ \t]*\)[ \t]*,?)+[ \t]*\]",
    re.IGNORECASE,
)
_BARE_CITATION = re.compile(
    r"[ \t]*PerQueryResult\([ \t]*index[ \t]*=[ \t]*"
    r"(?:'[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?)[ \t]*\)",
    re.IGNORECASE,
)
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,!?;:])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_PER_QUERY_WORD = "perqueryresult"
_COMPLETE_RESULT = re.compile(
    r"PerQueryResult\([ \t]*index[ \t]*=[ \t]*"
    r"(?:'[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?)[ \t]*\)[ \t]*$",
    re.IGNORECASE,
)
_UNCLOSED_RESULT = re.compile(r"PerQueryResult\([^)]*$", re.IGNORECASE)


def strip_file_search_citations(text: str) -> str:
    """Remove completed File Search citation markers from spoken text."""
    cleaned = _COMPLETE_CITATION.sub("", text)
    cleaned = _BARE_CITATION.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    return _MULTI_SPACE.sub(" ", cleaned)


def _open_bracket_is_citation(body: str) -> bool:
    """True when an unclosed '[' tail is empty or only a partial citation list."""
    if body.strip() == "":
        return True
    parts = body.split(",")
    for part in parts[:-1]:
        if _COMPLETE_RESULT.match(part.strip()) is None:
            return False
    tail = parts[-1].strip()
    if tail == "" or _COMPLETE_RESULT.match(tail) or _UNCLOSED_RESULT.match(tail):
        return True
    lower = tail.lower()
    return _PER_QUERY_WORD.startswith(lower)


def _citation_hold_index(text: str) -> int | None:
    """Return where an unfinished citation starts, if the tail must wait."""
    bracket = text.rfind("[")
    if bracket != -1 and "]" not in text[bracket + 1 :]:
        if _open_bracket_is_citation(text[bracket + 1 :]):
            start = bracket
            while start > 0 and text[start - 1] in " \t":
                start -= 1
            return start

    bare = re.search(r"PerQueryResult\([^)]*$", text, re.IGNORECASE)
    if bare:
        return bare.start()

    lower = text.lower()
    for length in range(len(_PER_QUERY_WORD), 3, -1):
        if lower.endswith(_PER_QUERY_WORD[:length]):
            return len(text) - length
    return None


def split_spoken_citations(text: str) -> tuple[str, str]:
    """Split text into a safe prefix and an unfinished citation suffix."""
    cleaned = strip_file_search_citations(text)
    hold_at = _citation_hold_index(cleaned)
    if hold_at is None:
        return cleaned, ""
    return cleaned[:hold_at], cleaned[hold_at:]


class SpokenCitationFilter(FrameProcessor):
    """Drop Gemini File Search citation markers before they are spoken."""

    def __init__(self):
        super().__init__()
        self._pending = ""

    async def _flush_pending(self) -> None:
        if not self._pending:
            return
        text = strip_file_search_citations(self._pending)
        self._pending = ""
        if _citation_hold_index(text) is not None:
            return
        if text:
            await self.push_frame(LLMTextFrame(text=text))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self._pending = ""
            await self.push_frame(frame, direction)
            return

        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, LLMTextFrame):
            if isinstance(frame, LLMFullResponseEndFrame):
                await self._flush_pending()
            await self.push_frame(frame, direction)
            return

        emit, self._pending = split_spoken_citations(self._pending + frame.text)
        if emit:
            frame.text = emit
            await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport):
    runner = WorkerRunner(handle_sigint=False)

    sarvam_api_key = os.environ["SARVAM_API_KEY"]
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
    language = os.getenv("SARVAM_LANGUAGE", "en-IN")

    stt = SarvamSTTService(
        api_key=sarvam_api_key,
        mode="translit",
        settings=SarvamSTTService.Settings(
            model=os.getenv("SARVAM_STT_MODEL", "saaras:v3"),
            language=language,
            vad_signals=False,
        ),
    )

    tts = SarvamTTSService(
        api_key=sarvam_api_key,
        settings=SarvamTTSService.Settings(
            model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
            voice=os.getenv("SARVAM_TTS_VOICE", "shubh"),
            language=Language(language),
            pace=1.0,
        ),
    )

    file_search_store = load_file_search_store()
    llm = GoogleLLMService(
        api_key=google_api_key,
        settings=GoogleLLMService.Settings(
            model=os.getenv("GEMINI_LLM_MODEL", "gemini-3.8-flash"),
            temperature=0.3,
            max_tokens=700,
            system_instruction=(
                "You are an NCERT Science tutor for students in Classes 6 to 10. "
                "You must always consult the File Search tool before answering any "
                "science questions from class 6 to 10 to ground every science answer "
                "in the textbook passages retrieved from that tool. "
                "Use the student's class when known. If class matters but is not known,"
                "ask one short clarifying question. If the retrieved text does not support "
                "an answer, say so rather than guessing. Explain at the student's grade level "
                "and reply in the language they use with a mix of English words where appropriate "
                "in every turn, but keep the script Latin for all languages otherwise your response "
                "will be invalid. Keep answers brief, warm, and conversational because they are spoken "
                "aloud. Do not use markdown, bullets, emojis, equations with unreadable notation, "
                "filenames, raw relevance scores, or citation markers such as PerQueryResult. "
            ),
            # File Search runs inside Gemini. Automatic function calling is the
            # SDK loop for Python callables, and it warns on generate_content_stream.
            extra={"automatic_function_calling": {"disable": True}},
        ),
    )
    context = LLMContext(
        tools=ToolsSchema(
            standard_tools=[],
            custom_tools={
                AdapterType.GEMINI: [
                    {
                        "file_search": {
                            "file_search_store_names": [file_search_store],
                        }
                    }
                ]
            },
        ),
    )
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
        ),
    )
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    speech_gate = SpeechAudioGate()
    citation_filter = SpokenCitationFilter()

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            speech_gate,
            stt,
            aggregators.user(),
            llm,
            citation_filter,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    agent = PipelineWorker(
        pipeline,
        name="ncert-science-tutor",
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @agent.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the student briefly as their NCERT Science tutor and "
                    "ask their class and what they want help with."
                ),
            }
        )
        await agent.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(agent)
    await runner.run()


async def start_bot(connection: SmallWebRTCConnection) -> None:
    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        webrtc_connection=connection,
    )
    await run_bot(transport)
