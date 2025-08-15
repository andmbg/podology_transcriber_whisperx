# Use CUDA runtime compatible with driver 12.0+
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /transcriber

# Set timezone to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies for WhisperX and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    sox \
    libsox-fmt-all \
    git \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHON_VERSION=3.11
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
RUN python3.11 -m venv /opt/venv

# Upgrade pip
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

RUN /opt/venv/bin/pip install \
    fastapi==0.115.12 \
    whisperx==3.3.4 \
    python-dotenv==1.1.0 \
    uvicorn==0.34.2 \
    python-multipart==0.0.20 \
    loguru==0.7.3

# Install PyTorch compatible with CUDA runtime 12.1 and driver 12.0+
RUN /opt/venv/bin/pip install \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify cuDNN 8 libraries are available (required by WhisperX)
RUN ldconfig && \
    find /usr -name "*cudnn*" -type f 2>/dev/null | head -5 && \
    echo "cuDNN libraries found" || echo "WARNING: cuDNN libraries not found"

# Copy application code
COPY podology_transcriber ./podology_transcriber

# Create necessary directories
RUN mkdir -p /transcriber/logs /transcriber/podology_transcriber/data

# Set GPU-optimized environment variables (made flexible for different runtimes)
# Don't set CUDA_VISIBLE_DEVICES - let the runtime handle this
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

# Use cuDNN 8 libraries from the base image (required by WhisperX)
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
ENV CUDNN_VERSION=8

# WhisperX specific optimizations
ENV WHISPERX_CACHE_DIR=/transcriber/.cache/whisperx
ENV HF_HOME=/transcriber/.cache/huggingface
ENV TORCH_HOME=/transcriber/.cache/torch

# API configuration
ENV API_TOKEN=""
ENV HF_TOKEN=""
ENV UVICORN_WORKERS=1
ENV UVICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker

# Expose the port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

    # Run with optimized settings for GPU transcription and large file uploads
    CMD ["/opt/venv/bin/uvicorn", "podology_transcriber.server:app", \
    "--host", "0.0.0.0", \
    "--port", "8001", \
     "--workers", "1", \
     "--timeout-keep-alive", "300", \
     "--limit-max-requests", "1000", \
     "--limit-concurrency", "10"]
