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

class LLMQA:

    def __init__(self, mode,LLM_name,path):
        self.mode = mode
        self.LLM_name = LLM_name
        self.path = path
        # 连接数据库
        conn = sqlite3.connect('db/md5.db')
        self.c = conn.cursor()
        # 创建表格
        self.c.execute('''CREATE TABLE IF NOT EXISTS files
                    (name text, md5 text)''')
    # 1.读取路径中的文件到documents
    def loadfiles(self):
        #计算文件md5
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            if os.path.isfile(file_path):
                # 计算文件的MD5值
                with open(file_path, 'rb') as f:
                    md5 = hashlib.md5()
                    while True:
                        data = f.read(4096)
                        if not data:
                            break
                        md5.update(data)
                    md5_value = md5.hexdigest()
            # 检查MD5值是否已存在于数据库中
            self.c.execute("SELECT * FROM files WHERE name=? AND md5=?", (filename, md5_value))
            data = self.c.fetchone()
            if data is None:
                # 如果MD5值不在数据库中，则将其添加到数据库中
                self.c.execute("INSERT INTO files VALUES (?, ?)", (filename, md5_value))
            else:
                # 如果MD5值已在数据库中，则更新对应的记录
                self.c.execute("UPDATE files SET md5=? WHERE name=?", (md5_value, filename))
            
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
    def createIndex(self,chunk_size_limit=512,index_type="openai"):
        match index_type:
            case "openai":
                self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, chunk_size_limit=512)
                self.index = GPTSimpleVectorIndex.from_documents(self.documents, service_context=self.service_context)
                self.index.save_to_disk("index_files/index.json")
                logging.info("索引消耗了%d个token",self.llm_predictor.last_token_usage)
            case "milvus":
                self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese"))
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
                    self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese"))
                    self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model,llm_predictor=self.llm_predictor, chunk_size_limit=512)
                    self.index = GPTSimpleVectorIndex.load_from_disk(filename)
        else:
            self.createIndex(index_type=index_type)
    
    def query(self,query):
        evaluator = ResponseEvaluator(service_context=self.service_context)
        QA_PROMPT_TMPL = (
            "我们有以下信息. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "请根据以上信息，只用中文回答问题: {query_str}\n"
        )
        QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
        response = self.index.query(
        query, 
        text_qa_template=QA_PROMPT,
        service_context=self.service_context,
        similarity_top_k=3
        ) 
        logging.info("查询消耗了%d个token",self.llm_predictor.last_token_usage)
        eval_result = evaluator.evaluate_source_nodes(response)
        logging.info(response.source_nodes[0].source_text[:1000] + "...")
        logging.info(str(eval_result))

        return response  
