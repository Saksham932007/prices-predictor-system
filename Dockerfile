# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app_user

# Copy requirements first for better caching
COPY prices-predictor-system/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY prices-predictor-system/ /app/

# Create necessary directories
RUN mkdir -p logs results models data/processed data/raw

# Change ownership to app_user
RUN chown -R app_user:app_user /app

# Switch to non-root user
USER app_user

# Expose port for MLflow UI
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pandas as pd; print('Health check passed')" || exit 1

# Default command
CMD ["python", "run_pipeline.py"]

# Labels for metadata
LABEL maintainer="Saksham Kapoor <saksham932007@example.com>"
LABEL version="1.0"
LABEL description="House Price Prediction ML Pipeline"
LABEL org.opencontainers.image.source="https://github.com/Saksham932007/prices-predictor-system"