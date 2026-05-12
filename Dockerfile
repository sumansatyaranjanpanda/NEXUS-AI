FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "nexus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
