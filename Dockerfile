FROM python:3.11-slim

WORKDIR /app

# ---- Environment ----
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_PATH=/app/chroma_db

# ---- Install dependencies ----
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Copy backend code (must include data/ folder) ----
COPY backend/ .

# ---- Build RAG index at image build time ----
# Fail-open so deployment never breaks if data folder missing
RUN python build_index.py || echo "Index build skipped"

# ---- Expose port ----
EXPOSE 8000

# ---- Start server ----
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]