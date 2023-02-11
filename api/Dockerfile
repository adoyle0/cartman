# syntax-docker/dockerfile:1
FROM python:slim
WORKDIR /app
COPY requirements.txt requirements.txt
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD [ "uvicorn", "chatbots:api", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
