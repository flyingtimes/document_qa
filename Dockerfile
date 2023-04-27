FROM clarkchan/llm-runtime
RUN pip3 install openAI llama_index
RUN pip3 install cpm_kernels
RUN pip3 install sentence_transformers
WORKDIR /app
