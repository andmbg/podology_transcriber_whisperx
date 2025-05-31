# filepath: kf_wxtrans/Dockerfile
FROM python:3.12-slim

WORKDIR /wxtrans

RUN apt-get update && apt-get install -y --no-install-recommends \
ffmpeg \
git \
build-essential \
&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY .env .env
COPY wxtrans ./wxtrans

ENV PYTHONUNBUFFERED=1

CMD ["poetry", "run", "uvicorn", "wxtrans.wxtrans:app", "--host", "0.0.0.0", "--port", "8001"]
