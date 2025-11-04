FROM docker.io/library/python:3.10-slim

# Force rebuild - v2
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (NOT google_gemini.py!)
COPY app.py .
COPY models.py .
COPY templates ./templates

# Expose port
EXPOSE 7860

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]