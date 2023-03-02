# syntax-docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt requirements.txt
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python3", "main.py"]
