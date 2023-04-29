import re
import sqlite3
import hashlib
import os
import transformers
from chatglm_llm import GlmLLM
from llama_index import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.evaluation import ResponseEvaluator
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, GPTListIndex, SimpleDirectoryReader, LLMPredictor,  QuestionAnswerPrompt
from sc import ServiceContext
import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')

# from chatglm_llm import GlmLLM
# helper functions for chinese text seprator
MAX_TEXT_INLINE = 250


def trim_text(text):
    """
    Trim text
    @param text:
    @return:
    """
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    text = re.sub('\n+', '\n', text)

    return text


def limit_line_length(text):
    """
    Limit line length
    @param text:
    @return:
    """
    lines = []
    for line in text.split('\n'):
        chunks = [line[i:i+MAX_TEXT_INLINE] for i in range(0, len(line), MAX_TEXT_INLINE)]
        lines.extend(chunks)
    return '\n\n'.join(lines)


class LLMQA:

    def __init__(self, mode, LLM_name, path):
        self.mode = mode
        self.LLM_name = LLM_name
        self.path = path
        self.gpt2 = None
    # 1.读取路径中的文件到documents
    def loadfiles(self):
        self.documents = SimpleDirectoryReader(self.path).load_data()
        # fix problem at here https://github.com/jerryjliu/llama_index/issues/453
        #for document in self.documents:
        #    document.text = limit_line_length(trim_text(document.text))

    # 2.加载模型
    def loadmodel(self, temperature=0.7):
        logging.info("using %s model.", self.LLM_name)
        if self.LLM_name == 'openAI3.5':
            self.llm_predictor = LLMPredictor(
                llm=OpenAI(temperature=temperature))
        elif self.LLM_name == "openAI3":
            self.gpt2 = transformers.GPT2TokenizerFast.from_pretrained("/gpt2")
            self.llm_predictor = LLMPredictor(llm=GlmLLM())
            
            responses=[
                "Action: Python REPL\nAction Input: print(2 + 2)",
                "Final Answer: 4",
                "Final Answer: 3",
                "Final Answer: 2",
                "Final Answer: 1",
                "Final Answer: 0",
                "Final Answer: 9",
                "Final Answer: 412",
            ]
            #self.llm_predictor = LLMPredictor(llm=FakeListLLM(responses=responses))

    # 3.创建索引
    def createIndex(self, chunk_size_limit=512, index_type="openai"):
        if index_type == "openai":
            self.service_context = ServiceContext.from_defaults(
                llm_predictor=self.llm_predictor,chunk_size_limit=512)
            self.index = GPTSimpleVectorIndex.from_documents(
                self.documents, service_context=self.service_context)
            self.index.save_to_disk("index_files/index.json")
            logging.info("索引消耗了%d个token", self.llm_predictor.last_token_usage)
        elif index_type == "milvus":
            # No sentence-transformers model found with name 不影响使用
            self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
                model_name="/GanymedeNil_text2vec-large-chinese"),tokenizer=self.gpt2)
            self.service_context = ServiceContext.from_defaults(
                embed_model=self.embed_model, llm_predictor=self.llm_predictor,tokenizer=self.gpt2,chunk_size_limit=512)
            self.index = GPTSimpleVectorIndex.from_documents(
                self.documents, service_context=self.service_context)
            self.index.save_to_disk("index_files/milvus.json")
            logging.info("使用huggingface做索引")

    # 4.加载索引
    def loadIndex(self, index_type="openai"):
        if index_type == "openai":
            filename = "index_files/index.json"
        elif index_type == "milvus":
            filename = "index_files/milvus.json"
        if os.path.exists(filename):
            logging.info("从文件加载")
            if index_type == "openai":
                self.service_context = ServiceContext.from_defaults(
                    llm_predictor=self.llm_predictor, chunk_size_limit=512)
                self.index = GPTSimpleVectorIndex.load_from_disk(filename)
            elif index_type == "milvus":
                self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="/GanymedeNil_text2vec-large-chinese"),tokenizer=self.gpt2)
                self.service_context = ServiceContext.from_defaults(
                    embed_model=self.embed_model, llm_predictor=self.llm_predictor,tokenizer=self.gpt2,chunk_size_limit=512)
                self.index = GPTSimpleVectorIndex.load_from_disk(filename)
        else:
            self.createIndex(index_type=index_type)

    def query(self, query):
        evaluator = ResponseEvaluator(service_context=self.service_context)
        QA_PROMPT_TMPL = (
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "请用中文回答，只根据以上信息作答，找不到相关信息就回答找不到: {query_str} \n"
        )
        QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
        response = self.index.query(
            query,
            text_qa_template=QA_PROMPT,
            service_context=self.service_context,
            similarity_top_k=1,
            verbose=True
        )

        logging.info("查询消耗了%d个token", self.llm_predictor.last_token_usage)
        logging.info(self.service_context.llama_logger.get_logs())

        return response
