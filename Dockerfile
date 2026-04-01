FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu \
 && pip install --no-cache-dir -r /app/requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . /app

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "glossary_pipeline.web", "--host", "0.0.0.0", "--port", "8080"]
