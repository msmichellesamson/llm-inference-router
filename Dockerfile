# Multi-stage build for production LLM inference router
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r llmrouter && \
    useradd -r -g llmrouter -s /bin/false -c "LLM Router User" llmrouter

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=llmrouter:llmrouter src/ ./src/
COPY --chown=llmrouter:llmrouter pyproject.toml ./
COPY --chown=llmrouter:llmrouter requirements.txt ./

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/cache && \
    chown -R llmrouter:llmrouter /app

# Switch to non-root user
USER llmrouter

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REDIS_URL=redis://localhost:6379 \
    LOG_LEVEL=INFO \
    MODEL_CACHE_DIR=/app/models \
    METRICS_PORT=8001

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-config", "src/logging_config.json"]