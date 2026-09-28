# TalkToTables - container for Azure Container Apps (or any container host)
# Build after running `python -m src.load_data` so data/olist.duckdb exists.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY rag/ rag/
COPY eval/ eval/
COPY app.py .
COPY .streamlit/ .streamlit/
COPY sample_data/ sample_data/
COPY data/olist.duckdb data/olist.duckdb

# Download the embedding model and build the retrieval index at build time,
# so the container starts fast and needs no internet for retrieval.
RUN python -c "from src.retriever import Retriever; print(Retriever(backend='embeddings').backend)"

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
