FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for reliability
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY worker.py .

# Run the worker
CMD ["python", "worker.py"]
