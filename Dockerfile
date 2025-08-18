FROM python:3.11-slim

# system packages (faiss wheels should work; libgomp helps on some bases)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir faiss-cpu==1.8.0.post1 sentence-transformers==2.7.0 pypdf==4.2.0 httpx==0.27.0

COPY . /app

# ensure data dir exists (Render disk mounts to /data)
RUN mkdir -p /data
ENV PYTHONUNBUFFERED=1

# On first boot, app will ingest /data/PonnamcCV.pdf into /data/index.faiss
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
