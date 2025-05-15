from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple

# Modelo preentrenado de BioGPT (GPT2-like)
MODEL_NAME = "microsoft/BioGPT-Large"


def load_model_and_tokenizer(use_gpu: bool = False) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """
    Carga el modelo y el tokenizador de BioGPT.

    Args:
        use_gpu (bool): Indica si se debe utilizar GPU (True) o CPU (False).

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]: Tokenizador, modelo y dispositivo de cómputo.
    """
    print("📦 Cargando modelo BioGPT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ Modelo cargado en {'GPU' if device.type == 'cuda' else 'CPU'}.")

    return tokenizer, model, device


# Inicializar modelo y tokenizador
tokenizer, model, device = load_model_and_tokenizer(use_gpu=False)


def generate_biomedical_answer(prompt: str, max_new_tokens: int = 200) -> str:
    """
    Genera una respuesta biomédica utilizando el modelo BioGPT.

    Args:
        prompt (str): Pregunta o contexto inicial para la generación de texto.
        max_new_tokens (int): Número máximo de tokens generados en la respuesta.

    Returns:
        str: Respuesta generada por el modelo.
    """
    print("✍️ Generando respuesta...")
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    print("✅ Respuesta generada.")
    return response
