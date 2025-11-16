# -----------------------------
# Stage 1: Builder (for dependencies)
# -----------------------------
FROM python:3.10-slim AS builder

# Set work directory
WORKDIR /app

# Copy only requirements to leverage caching
COPY requirements.txt .

# Install build dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    python -m pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove git build-essential && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------
# Stage 2: Runtime (small final image)
# -----------------------------
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only application code
COPY . .

# Expose app port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

