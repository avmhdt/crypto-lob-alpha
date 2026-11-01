FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.12.23
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY config ./config
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
STOPSIGNAL SIGTERM
ENTRYPOINT ["lob", "record", "run", "--config", "config/recorder.yaml"]
