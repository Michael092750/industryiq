# Recipe to build an image of the IndustryIQ API.
FROM python:3.11-slim

# Don't write .pyc files; stream logs straight to the console.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Optional extras baked into the image. Empty by default so production (Bedrock)
# stays lean; docker-compose.override.yml sets EXTRAS=local for local dev, which
# pulls in fastembed -- the CPU embedder that RAG_PROVIDER=anthropic needs.
ARG EXTRAS=

# Install dependencies first (this layer is cached unless pyproject changes).
COPY pyproject.toml ./
COPY src/ ./src/
RUN if [ -n "$EXTRAS" ]; then pip install --no-cache-dir ".[$EXTRAS]"; else pip install --no-cache-dir .; fi

EXPOSE 8000

# How the container runs the app. 0.0.0.0 so it's reachable from outside the container.
CMD ["uvicorn", "industryiq.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
