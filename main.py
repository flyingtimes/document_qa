# query document
# 写一个restful api 接口
from documentLLMQuery import LLMQA
import logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s  %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S +0000')
if __name__=="__main__":
    logging.info("开始")
    a = LLMQA(mode='online',LLM_name='openAI3.5',path='txt')
    logging.info("加载文件")
    a.loadfiles()
    logging.info("加载模型")
    a.loadmodel()
    logging.info("加载索引")
    a.loadIndex(index_type="milvus")


    logging.info("开始查询")
    print("answer:"+str(a.query("哪些人反对司马光变法")))
