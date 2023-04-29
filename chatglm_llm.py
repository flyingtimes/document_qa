from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
from typing import Optional, List
import torch
class GlmLLM(LLM):
    model_name = "/model-dir"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True,trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True,trust_remote_code=True).float()
    history = []
    max_token = 512
    temperature = 0.7

    def _call(self,
        prompt: str,
        history: List[List[str]] = [],
        stop: Optional[List[str]] = None) -> str:
        
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        return response

    def _identifying_params(self):
        return {"name_of_model": self.model_name}
    
    @property
    def _llm_type(self):
        return "custom"
