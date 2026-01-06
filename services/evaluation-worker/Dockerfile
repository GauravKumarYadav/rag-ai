FROM python:3.11-slim

WORKDIR /app

# Configure pip for reliability
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY worker.py .

# Run the worker
CMD ["python", "worker.py"]
