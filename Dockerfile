FROM python:3.9-slim

# Install ffmpeg and other dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p uploads static/previews static/safe_versions static/reports && \
    chmod -R 777 uploads static

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Ensure proper permissions
RUN chmod -R 777 /app

# Command to run the application
CMD gunicorn app:app --timeout 300 --bind 0.0.0.0:$PORT 