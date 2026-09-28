"""Host the tutor on the eval transport for ``pipecat eval``.

The production entry points join Daily or WebRTC. This one hosts a local
WebSocket the eval harness connects to, and runs the same ``run_bot`` pipeline.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from pipecat.evals.serializer import EvalSerializer
from pipecat.evals.transport import EvalTransport, EvalTransportParams

from bot import run_bot


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tutor for Pipecat evals.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    params = EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    params.serializer = EvalSerializer()
    transport = EvalTransport(params=params, host=args.host, port=args.port)

    @transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(_transport):
        logger.info("Eval transport ready at ws://{}:{}", args.host, args.port)

    await run_bot(transport)


if __name__ == "__main__":
    asyncio.run(main())
