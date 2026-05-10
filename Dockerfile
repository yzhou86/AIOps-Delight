# syntax=docker/dockerfile:1

FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5005

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir gunicorn

COPY backend /app/backend
COPY examples /app/examples
COPY README.md /app/README.md
COPY --from=frontend-builder /build/frontend/dist /app/frontend/dist

RUN mkdir -p /app/backend/uploads /app/backend/data

EXPOSE 5005

CMD ["sh", "-c", "gunicorn --chdir backend --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-120} --bind 0.0.0.0:${PORT:-5005} app:app"]
