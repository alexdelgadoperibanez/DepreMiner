from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Cargar BioGPT (GPT2-like)
MODEL_NAME = "microsoft/BioGPT-Large"
#@st.cache_resource(show_spinner="🔄 Cargando modelo BioGPT...")
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    USE_GPU = False  # cambia a True si tienes suficiente VRAM

    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

def generate_biomedical_answer(prompt, max_new_tokens=200):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()