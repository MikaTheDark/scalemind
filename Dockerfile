# Image de service : l'API de recommandation, sans PyTorch (inférence numpy).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY scalemind ./scalemind
RUN pip install --no-cache-dir ".[serve]"

# seule la politique exportée est nécessaire (quelques Ko)
COPY models/policy.npz ./models/policy.npz

RUN useradd --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "scalemind.serve:app", "--host", "0.0.0.0", "--port", "8000"]
