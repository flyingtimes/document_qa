# query document
# 写一个restful api 接口
from documentLLMQuery import LLMQA
import logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s  %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S +0000')
if __name__=="__main__":
    print("程序开始加载...")
    a = LLMQA(mode='online',LLM_name='openAI3',path='txt')
    a.loadfiles()
    a.loadmodel()
    a.loadIndex(index_type="milvus")
    logging.info("加载完毕")
    while True:
        try:
            query = input("请输入问题:")
            if query=='stop':
                break
            elif query=='reload':
                print("重新加载索引")
                a.loadfiles()
                a.createIndex(index_type='milvus')
            else:
                print("thinking...")
                print("answer:"+str(a.query(query)))
        except:
            continue