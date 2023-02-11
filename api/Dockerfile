# syntax-docker/dockerfile:1
FROM alpine
WORKDIR /app
COPY requirements.txt requirements.txt
RUN apk add --no-cache --virtual build-deps py3-pip rust cargo g++ openssl-dev python3-dev
RUN pip install --no-cache-dir -r requirements.txt
RUN apk del build-deps
RUN rm -rf /var/cache/apk/*
COPY . .
CMD [ "uvicorn", "chatbots:api", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
