# Use NVIDIA CUDA base image optimized for PyTorch/ML workloads
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

WORKDIR /transcriber

# Install system dependencies for WhisperX and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
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
ENV PYTHON_VERSION=3.12
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
RUN python3.12 -m venv /opt/venv

# Upgrade pip
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Install dependencies directly with pip (much simpler than Poetry)
RUN /opt/venv/bin/pip install \
    fastapi==0.115.12 \
    whisperx==3.3.4 \
    python-dotenv==1.1.0 \
    uvicorn==0.34.2 \
    python-multipart==0.0.20 \
    loguru==0.7.3

# Install additional GPU-optimized dependencies
RUN /opt/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN /opt/venv/bin/pip install flash-attn --no-build-isolation

# Copy application code
COPY podology_transcriber ./podology_transcriber

# Create necessary directories
RUN mkdir -p /transcriber/logs /transcriber/podology_transcriber/data

# Set GPU-optimized environment variables (made flexible for different runtimes)
# Don't set CUDA_VISIBLE_DEVICES - let the runtime handle this
# ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

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

# Run with optimized settings for GPU transcription
CMD ["/opt/venv/bin/uvicorn", "podology_transcriber.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]
