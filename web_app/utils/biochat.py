import re
from typing import Tuple, Optional, List

import torch
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from web_app.utils.bio_chat import load_model_and_tokenizer


class BioGPTLLM(LLM):
    _tokenizer: AutoTokenizer = PrivateAttr()
    _model: AutoModelForCausalLM = PrivateAttr()
    _device: torch.device = PrivateAttr()

    def __init__(self, tokenizer, model, device, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        input_ids = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id
            )

        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

    @property
    def _llm_type(self) -> str:
        return "biogpt"

# Prompt para LangChain
bio_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
Clinical question: {question}

Based on the abstracts:
{context}

Answer:"""
)

# Carga de modelos fuera de la función
tokenizer, gpt_model, device = load_model_and_tokenizer(use_gpu=False)
qa_pipeline = pipeline("question-answering", model="emilyalsentzer/Bio_ClinicalBERT")

def generate_biomedical_answer_langchain(question: str, context: str) -> Tuple[str, str]:
    """
    Devuelve dos respuestas: una con Bio_ClinicalBERT y otra con BioGPT usando LangChain.
    """
    result = qa_pipeline(question=question, context=context)
    respuesta_qa = result["answer"].strip()

    # Fallback si la respuesta es pobre
    needs_fallback = (
        not respuesta_qa
        or len(respuesta_qa.split()) < 10
        or not re.search(r"\b\w+ed\b", respuesta_qa.lower())
    )

    biogpt_response = ""
    if needs_fallback:
        biogpt_llm = BioGPTLLM(tokenizer=tokenizer, model=gpt_model, device=device)
        chain = LLMChain(llm=biogpt_llm, prompt=bio_prompt)
        biogpt_response = chain.run(question=question, context=context)

    return respuesta_qa, biogpt_response
