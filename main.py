"""FastAPI app for the NCERT science voice tutor."""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from loguru import logger
from pipecat_ai_prebuilt.frontend import PipecatPrebuiltUI

from pipecat.runner.daily import configure
from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomProperties
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import start_bot, start_daily_bot

# One student plus the tutor. Rooms expire so a leftover link cannot be reused.
_DAILY_SESSION_SECONDS = 60 * 60

webrtc_handler = SmallWebRTCRequestHandler()
bot_tasks: set[asyncio.Task] = set()
active_sessions: set[str] = set()


def _bot_finished(task: asyncio.Task) -> None:
    bot_tasks.discard(task)
    if not task.cancelled() and (error := task.exception()):
        logger.error("Voice bot session failed: {}", error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await webrtc_handler.close()
    for task in bot_tasks:
        task.cancel()
    if bot_tasks:
        await asyncio.gather(*bot_tasks, return_exceptions=True)


app = FastAPI(
    title="NCERT Science Voice Tutor",
    description="A WebRTC voice tutor powered by Sarvam, Gemini, and NCERT File Search.",
    lifespan=lifespan,
)
app.mount("/client", PipecatPrebuiltUI, name="client")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/client/")


@app.get("/call", include_in_schema=False)
async def start_daily_call():
    """Create a private voice-only Daily room, join the tutor, and send the browser there."""
    api_key = os.getenv("DAILY_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Set DAILY_API_KEY in .env to the Daily key from your Pipecat Cloud account.",
        )
    try:
        import daily  # noqa: F401
    except ImportError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Daily's Python library has no Windows build. "
                "Run this server with Docker so the tutor can join the room."
            ),
        ) from error

    try:
        async with aiohttp.ClientSession() as session:
            room = await configure(
                session,
                api_key=api_key,
                room_properties=DailyRoomProperties(
                    exp=time.time() + _DAILY_SESSION_SECONDS,
                    eject_at_room_exp=True,
                    max_participants=2,
                    start_video_off=True,
                    enable_prejoin_ui=False,
                ),
            )
            student_token = await DailyRESTHelper(
                daily_api_key=api_key,
                aiohttp_session=session,
            ).get_token(
                room.room_url,
                expiry_time=_DAILY_SESSION_SECONDS,
                owner=False,
            )
    except Exception as error:
        logger.error("Failed to create Daily room: {}", error)
        raise HTTPException(
            status_code=502,
            detail="Daily could not create a room. Check DAILY_API_KEY.",
        ) from error

    task = asyncio.create_task(
        start_daily_bot(room.room_url, room.token),
        name=f"daily-bot-{room.room_url.rstrip('/').rsplit('/', 1)[-1]}",
    )
    bot_tasks.add(task)
    task.add_done_callback(_bot_finished)
    logger.info("Tutor joining Daily room {}", room.room_url)
    return RedirectResponse(url=f"{room.room_url}?t={student_token}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/start")
async def start_agent(request: Request):
    """Create a WebRTC session. The client then calls /sessions/{id}/api/offer."""
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    transport = request_data.get("transport") or "webrtc"
    if transport != "webrtc":
        raise HTTPException(
            status_code=400,
            detail=f"Transport '{transport}' is not supported. Use webrtc.",
        )

    session_id = str(uuid.uuid4())
    active_sessions.add(session_id)

    result: dict = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    return result


@app.api_route("/sessions/{session_id}/api/offer", methods=["POST", "PATCH"])
async def session_offer(session_id: str, request: Request):
    if session_id not in active_sessions:
        return Response(content="Invalid or not-yet-ready session_id", status_code=404)

    try:
        request_data = await request.json()
    except Exception as error:
        logger.error("Failed to parse WebRTC request: {}", error)
        return Response(content="Invalid WebRTC request", status_code=400)

    if request.method == "POST":
        return await offer(
            SmallWebRTCRequest(
                sdp=request_data["sdp"],
                type=request_data["type"],
                pc_id=request_data.get("pc_id"),
                restart_pc=request_data.get("restart_pc"),
            )
        )

    return await ice_candidate(
        SmallWebRTCPatchRequest(
            pc_id=request_data["pc_id"],
            candidates=[
                IceCandidate(**candidate)
                for candidate in request_data.get("candidates", [])
            ],
        )
    )


async def offer(request: SmallWebRTCRequest):
    async def on_connection(connection: SmallWebRTCConnection) -> None:
        task = asyncio.create_task(
            start_bot(connection),
            name=f"voice-bot-{connection.pc_id}",
        )
        bot_tasks.add(task)
        task.add_done_callback(_bot_finished)

    return await webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )


async def ice_candidate(request: SmallWebRTCPatchRequest):
    await webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "7860")),
    )
