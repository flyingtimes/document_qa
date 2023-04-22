import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s  %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S +0000')

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, QuestionAnswerPrompt
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from llama_index.evaluation import ResponseEvaluator
import os
class LLMQA:

    def __init__(self, mode,LLM_name,path):
        self.mode = mode
        self.LLM_name = LLM_name
        self.path = path
    # 1.读取路径中的文件到documents
    def loadfiles(self):
        self.documents = SimpleDirectoryReader(self.path).load_data()

    # 2.加载模型
    def loadmodel(self,temperature=0.7):
        logging.info("using %s model.",self.LLM_name)
        match self.LLM_name:
            case "openAI3.5":
                self.llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo"))
            case "openAI3":
                self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=temperature, model_name="text-ada-001"))

    # 3.创建索引
    def createIndex(self,chunk_size_limit=512):
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, chunk_size_limit=512)
        self.index = GPTSimpleVectorIndex.from_documents(self.documents, service_context=self.service_context)
        self.index.save_to_disk("index.json")
        logging.info("消耗了%d个token",self.llm_predictor.last_token_usage)
        match self.LLM_name:
            case "openAI3.5":
                logging.info("费用为%d元",self.llm_predictor.last_token_usage/1000*0.002*7)
            case "openAI3":
                logging.info("费用为%d元",self.llm_predictor.last_token_usage/1000*0.0004*7)    

    # 4.加载索引
    def loadIndex(self):
        if os.path.exists("index.json"):
            self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, chunk_size_limit=512)
            #self.index = GPTSimpleVectorIndex.from_documents(self.documents)
            #self.index = GPTSimpleVectorIndex()
            self.index=GPTSimpleVectorIndex.load_from_disk('index.json')
        else:
            self.createIndex()
    
    def query(self,query):
        evaluator = ResponseEvaluator(service_context=self.service_context)
        QA_PROMPT_TMPL = (
            "我们有以下信息. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "请根据以上信息回答问题: {query_str}\n"
        )
        QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
        response = self.index.query(
        query, 
        text_qa_template=QA_PROMPT,
        service_context=self.service_context,
        similarity_top_k=3
        ) 
        logging.info("消耗了%d个token",self.llm_predictor.last_token_usage)
        match self.LLM_name:
            case "openAI3.5":
                logging.info("费用为%d元",self.llm_predictor.last_token_usage/1000*0.002*7)
            case "openAI3":
                logging.info("费用为%d元",self.llm_predictor.last_token_usage/1000*0.0004*7)
        eval_result = evaluator.evaluate_source_nodes(response)
        logging.info(response.source_nodes[0].source_text[:1000] + "...")
        logging.info(str(eval_result))

        return response  
