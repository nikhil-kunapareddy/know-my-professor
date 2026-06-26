# Single Dockerfile for all five Cloud Run deployables.
# Built once per unit with a different COMPONENT (profiles | weblinks | ingest |
# api | frontend); each install pulls ONLY that component's dependency extra.
#
#   docker build -f deploy/Dockerfile --build-arg COMPONENT=ingest -t kmp-ingest .
#
# Build context is the REPO ROOT (so the whole package tree is available).
FROM python:3.12-slim

ARG COMPONENT
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    COMPONENT=${COMPONENT}

WORKDIR /app

# Install dependencies first (better layer caching), then the source.
COPY pyproject.toml ./
COPY core ./core
COPY preprocessing ./preprocessing
COPY shared ./shared
COPY serving ./serving
COPY entrypoint.py ./
RUN pip install ".[${COMPONENT}]"

# Services listen on $PORT; Jobs ignore it. The dispatcher routes by $COMPONENT.
CMD ["python", "entrypoint.py"]
