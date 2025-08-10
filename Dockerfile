# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    HF_HOME=/app/cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire app directory
COPY app/ .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "8000"]