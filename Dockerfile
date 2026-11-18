FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Non-root user (uid 1000 also satisfies Hugging Face Spaces)
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data/processed ./data/processed
RUN mkdir -p data/index && chown -R appuser:appuser /home/appuser

USER appuser

# Bake the embedding model + vector index into the image so start-up is fast and offline-safe
RUN python -c "from app.rag import Retriever; r = Retriever(); print('index ready:', len(r.chunks), 'chunks, dense =', r.emb is not None)"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:'+__import__('os').environ.get('PORT','7860')+'/api/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
