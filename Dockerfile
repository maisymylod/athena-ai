FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first — much smaller wheel than the default GPU build.
RUN pip install --no-cache-dir \
        torch==2.5.* torchvision==0.20.* \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY ml/ ./ml/
COPY server/ ./server/
COPY tools/ ./tools/
COPY static/ ./static/
COPY index.html ./
# Optional: bake a checkpoint into the image if present at build time.
# In practice we mount checkpoints as a volume in production.
COPY checkpoints/ ./checkpoints/

ENV PORT=8080 \
    ATHENA_CHECKPOINT=/app/checkpoints/best_model.pt
EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", \
     "-b", "0.0.0.0:8080", "--timeout", "60", \
     "server.app:create_app()"]
