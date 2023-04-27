from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
from typing import Optional, List
import torch
class GlmLLM(LLM):
    model_name = "/model-dir"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True,trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True,trust_remote_code=True).half().quantize(4).cuda()
    history = []

    def _call(self,
        prompt: str,
        history: List[List[str]] = [],
        stop: Optional[List[str]] = None) -> str:
        
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        return response, history

    def _identifying_params(self):
        return {"name_of_model": self.model_name}
    
    @property
    def _llm_type(self):
        return "custom"
