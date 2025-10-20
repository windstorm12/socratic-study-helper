FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY google_gemini.py .
COPY templates/ templates/

EXPOSE 7860

# Use Gunicorn instead of Flask dev server
CMD ["gunicorn", "-w", "1", "-k", "sync", "--timeout", "120", "-b", "0.0.0.0:7860", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "google_gemini:app"]