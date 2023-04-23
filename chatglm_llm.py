from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
class GlmLLM(LLM):
    model_name = "./models/chatglm"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    history = []
    def _call(self, prompt, stop=None):
        response, self.history = self.model.chat(self.tokenizer, prompt, history=self.history)
        return response

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"