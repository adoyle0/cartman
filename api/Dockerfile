# syntax-docker/dockerfile:1
FROM pytorch/pytorch
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "uvicorn", "chatbots:api", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
