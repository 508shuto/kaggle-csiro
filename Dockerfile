FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git git-lfs unzip \
    python3 python3-dev python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

CMD ["uv", "run", "jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--allow-root"]

