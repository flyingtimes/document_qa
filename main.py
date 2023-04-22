# query document
# 写一个restful api 接口
from documentLLMQuery import LLMQA

a = LLMQA(mode='online',LLM_name='openAI3.5',path='txt')
print("加载文件")
a.loadfiles()
print("加载模型")
a.loadmodel()
print("加载索引")
a.loadIndex()
print("开始查询")
print(a.query("OSD中一个100G的文件会切割成多少个object"))