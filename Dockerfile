# filepath: kf_wxtrans/Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY kf_wxtrans ./kf_wxtrans

ENV PYTHONUNBUFFERED=1

CMD ["poetry", "run", "uvicorn", "kf_wxtrans.wxtrans:app", "--host", "0.0.0.0", "--port", "8001"]
