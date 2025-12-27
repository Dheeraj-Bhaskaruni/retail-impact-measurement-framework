FROM python:3.10-slim

LABEL maintainer="Dheeraj Bhaskaruni <dheeraj58799@gmail.com>"
LABEL description="Retail Promotion Impact Measurement Pipeline"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code
COPY config/ config/
COPY src/ src/
COPY data/synthetic/ data/synthetic/

# Generate dev data if not mounted
RUN python data/synthetic/generate_data.py

# Default: run the measurement pipeline
WORKDIR /app/src
CMD ["python", "-m", "pipeline.measurement_pipeline"]
