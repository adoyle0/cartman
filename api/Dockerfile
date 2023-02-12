# syntax-docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt requirements.txt
ENV PYTHONUNBUFFERED=1
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "uvicorn", "chatbots:api", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
