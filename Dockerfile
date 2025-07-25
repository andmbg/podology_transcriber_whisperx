FROM superlinear/python-gpu:3.12-cuda11.8

WORKDIR /transcriber

RUN apt-get update && apt-get install -y --no-install-recommends \
ffmpeg \
git \
build-essential \
&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY podology_transcriber ./podology_transcriber

# Set default environment variables (can be overridden in vast.ai)
ENV API_TOKEN=""
ENV HF_TOKEN=""

# Expose the port
EXPOSE 8001

CMD ["poetry", "run", "uvicorn", "podology_transcriber.server:app", "--host", "0.0.0.0", "--port", "8001"]
