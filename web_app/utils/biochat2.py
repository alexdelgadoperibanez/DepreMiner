import re

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple

# === Modelo 1: Bio_ClinicalBERT para QA extractiva ===
qa_pipeline = pipeline("question-answering", model="emilyalsentzer/Bio_ClinicalBERT")

# === Modelo 2: BioGPT para generación libre ===
MODEL_NAME = "microsoft/BioGPT-Large"

def load_model_and_tokenizer(use_gpu: bool = False) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    print("📦 Cargando modelo BioGPT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ Modelo cargado en {'GPU' if device.type == 'cuda' else 'CPU'}.")
    return tokenizer, model, device

# Inicializar BioGPT solo una vez
tokenizer, gpt_model, device = load_model_and_tokenizer(use_gpu=False)


def generate_biomedical_answer(question: str, context: str, max_new_tokens: int = 200) -> str:
    """
    Intenta responder usando Bio_ClinicalBERT (QA). Si falla, recurre a BioGPT.

    Args:
        question (str): Pregunta médica.
        context (str): Abstractos concatenados.
        max_new_tokens (int): Tokens para generación en fallback.

    Returns:
        str: Respuesta clínica.
    """
    print("🔎 Intentando respuesta con Bio_ClinicalBERT...")
    result = qa_pipeline(question=question, context=context)

    answer = result["answer"].strip()
    # Fallback si la respuesta es demasiado corta o no informativa
    if not answer or len(answer.split()) < 10 or not re.search(r"\b\w+ed\b", answer.lower()):
        print("⚠️ Respuesta poco informativa. Recurriendo a BioGPT...")
        prompt = f"""Clinical question: {question}\n\nBased on the abstracts:\n{context}\n\nAnswer:"""
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)

        with torch.no_grad():
            outputs = gpt_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        #return response

    print("✅ Respuesta generada por QA.")
    return answer, response
