FROM python:3.10
RUN pip3 install openAI llama_index
RUN pip3 install pymilvus
RUN pip3 install sentence_transformers
WORKDIR /app