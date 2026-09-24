# Use a slim Python image
FROM python:3.11-slim

# Install system dependencies (minimal for testing)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the drivers you'll need for Aiven
# Create a requirements.txt with: confluent-kafka, psycopg2-binary, redis
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your current folders (Utilities, stock_analysis, etc.)
COPY . .

# KEEP-ALIVE: This prevents the container from exiting immediately
CMD ["tail", "-f", "/dev/null"]