# syntax-docker/dockerfile:1
FROM python:3.10-slim
COPY . .
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python3", "main.py"]
EXPOSE 6969
