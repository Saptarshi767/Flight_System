# Multi-stage build for production optimization
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Development stage
FROM base as development

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with dev tools
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env.example .env
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Create necessary directories
RUN mkdir -p data logs exports reports && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy requirements first for better caching
COPY requirements.txt .

# Install only production dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Copy application code (excluding dev files)
COPY src/ ./src/
COPY .env.example .env

# Create necessary directories
RUN mkdir -p data logs exports reports && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check with more robust checking
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with gunicorn
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-"]