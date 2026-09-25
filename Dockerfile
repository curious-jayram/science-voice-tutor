FROM dailyco/pipecat-base:latest-py3.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The Windows CLI uploads nested files with backslashes in the name.
# Turn those into a real rag package. A Linux upload already has the directory.
COPY . /tmp/ctx
RUN python - <<'PY'
from pathlib import Path

root = Path("/tmp/ctx")
for path in list(root.iterdir()):
    if path.is_file() and "\\" in path.name:
        dest = root.joinpath(*path.name.split("\\"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        path.rename(dest)
PY
RUN test -f /tmp/ctx/rag/config.py && cp /tmp/ctx/bot.py ./bot.py && cp -a /tmp/ctx/rag ./rag
