# dockerfiles/api.dockerfile
FROM python:3.11-slim

WORKDIR /app

# Ensure system has basic build tools for faiss, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for caching
COPY pyproject.toml requirements.txt ./

# Install project (this will read requirements.txt via pyproject.toml)
RUN pip install --no-cache-dir .

# Copy source code & models
COPY src ./src
COPY models ./models

# Environment variables
ENV PYTHONPATH=/app/src
ENV OTTO_RECSYS_MODEL_RUN=1.0__df3a9d30

EXPOSE 8000

# Uvicorn entrypoint — use the installed package, NOT src. path
CMD ["uvicorn", "ottorecsys.api:app", "--host", "0.0.0.0", "--port", "8000"]


