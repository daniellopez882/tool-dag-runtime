# syntax=docker/dockerfile:1

# ---- build: resolve the locked dependencies and install the package into /opt/venv ----
FROM python:3.12-slim AS build

# uv is pinned to the release that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# ---- runtime: the virtualenv on the same base image, no build tools, non-root ----
FROM python:3.12-slim AS runtime

RUN groupadd --gid 10001 tooldag \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin tooldag \
    && mkdir -p /data \
    && chown 10001:10001 /data

COPY --from=build /opt/venv /opt/venv

# Without fetched data under /data the service falls back to the bundled fixture
# catalogue; mount a directory filled by `tooldag fetch` to use the real one.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TOOLDAG_DATA_DIR=/data

USER 10001:10001
WORKDIR /home/tooldag
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2); sys.exit(0 if r.status == 200 else 1)"]

ENTRYPOINT ["tooldag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
