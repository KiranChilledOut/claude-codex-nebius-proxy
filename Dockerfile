FROM ghcr.io/astral-sh/uv:bookworm-slim

# Install build tools required for httptools / uvicorn
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date.
# The bookworm-slim image bundles no python3, so uv downloads a managed CPython.
# Pin to 3.12 (langfuse v4 requires >=3.10; 3.9 is no longer supported).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
RUN uv python install 3.12 && uv sync --locked --python 3.12

# Start proxy
CMD ["uv", "run", "start_proxy.py"]
