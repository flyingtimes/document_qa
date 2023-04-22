FROM python:3.10
RUN pip3 install openAI llama_index
WORKDIR /app