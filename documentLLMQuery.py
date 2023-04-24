import logging
import sys

#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s  %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S +0000')

from llama_index import GPTSimpleVectorIndex, GPTListIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, QuestionAnswerPrompt
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from llama_index.evaluation import ResponseEvaluator
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
import os
import hashlib
import sqlite3
#from chatglm_llm import GlmLLM

class LLMQA:

    def __init__(self, mode,LLM_name,path):
        self.mode = mode
        self.LLM_name = LLM_name
        self.path = path
       
    # 1.读取路径中的文件到documents
    def loadfiles(self):
            
        self.documents = SimpleDirectoryReader(self.path).load_data()
        #from llama_index import download_loader

        #BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader",custom_path="/app/hubtools")
        #loader = BeautifulSoupWebReader()
        #self.documents = loader.load_data(urls=['https://news.cctv.com/2023/03/15/ARTIfHlCsQPehxnwHaklLUqP230315.shtml'])
        #self.documents = loader.load_data(urls=['https://www.msdmanuals.cn/home/disorders-of-nutrition/minerals/wilson-disease'])

    # 2.加载模型
    def loadmodel(self,temperature=0.7):
        logging.info("using %s model.",self.LLM_name)
        match self.LLM_name:
            case "openAI3.5":
                self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=temperature))
            case "openAI3":
                self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=temperature, model_name="text-ada-001"))

    # 3.创建索引
    def createIndex(self,chunk_size_limit=512,index_type="openai"):
        match index_type:
            case "openai":
                self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, chunk_size_limit=512)
                self.index = GPTSimpleVectorIndex.from_documents(self.documents, service_context=self.service_context)
                self.index.save_to_disk("index_files/index.json")
                logging.info("索引消耗了%d个token",self.llm_predictor.last_token_usage)
            case "milvus":
                self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="/root/.cache/torch/sentence_transformers/GanymedeNil_text2vec-large-chinese"))
                #self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese"))
                #self.service_context = ServiceContext.from_defaults(embed_model=embed_model)

                self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm_predictor=self.llm_predictor, chunk_size_limit=512)
                self.index = GPTSimpleVectorIndex.from_documents(self.documents, service_context=self.service_context)
                #elf.index = GPTSimpleVectorIndex.from_documents(self.documents, host="milvus", overwrite=True)
                #self.index = GPTListIndex(self.documents, embed_model=embed_model)
                self.index.save_to_disk("index_files/milvus.json")
                logging.info("使用huggingface做索引")

    # 4.加载索引
    def loadIndex(self,index_type="openai"):
        match index_type:
            case "openai":
                filename = "index_files/index.json"
            case "milvus":
                filename = "index_files/milvus.json"
        if os.path.exists(filename):
            match index_type:
                case "openai":
                    self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, chunk_size_limit=512)
                    #self.index = GPTSimpleVectorIndex.from_documents(self.documents)
                    #self.index = GPTSimpleVectorIndex()
                    self.index= GPTSimpleVectorIndex.load_from_disk(filename)
                case "milvus":
                    self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="/root/.cache/torch/sentence_transformers/GanymedeNil_text2vec-large-chinese"))
             #       self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese"))
                    self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model,llm_predictor=self.llm_predictor, chunk_size_limit=512)
                    self.index = GPTSimpleVectorIndex.load_from_disk(filename)
        else:
            self.createIndex(index_type=index_type)
    
    def query(self,query):
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
        similarity_top_k=5,
        verbose=True
        ) 
        logging.info("查询消耗了%d个token",self.llm_predictor.last_token_usage)
        formatted_res = response.get_formatted_sources()
        eval_result = evaluator.evaluate_source_nodes(response)
        i=0
        for rs in eval_result:
            if rs == "YES":
                logging.info(str(formatted_res[i]))
            i=i+1

        logging.info(response.get_formatted_sources())
        #logging.info(response.source_nodes[0].source_text[:1000] + "...")
        logging.info(str(eval_result))

        return response  
