# filepath: kf_wxtrans/Dockerfile
FROM python:3.12-slim

WORKDIR /wxtrans

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY wxtrans ./wxtrans

ENV PYTHONUNBUFFERED=1

CMD ["poetry", "run", "uvicorn", "wxtrans.wxtrans:app", "--host", "0.0.0.0", "--port", "8001"]
