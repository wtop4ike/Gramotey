import torch
import os
import re 
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk

MODEL_DIR = "./final_word_model_test_run" 
XLA_AVAILABLE = False
try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    pass

if XLA_AVAILABLE:
    device = torch.device("xla:0")
    print("> TPU/XLA обнаружен. Инференс будет использовать XLA-бэкэнд.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"> CUDA обнаружена. Используемое устройство: {device}")
else:
    device = torch.device("cpu")
    print(f"> Используемое устройство для инференса: {device}")

try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, safe_serialization=False)
    model.to(device)
    model.eval()
    print(f"> Модель и токенайзер успешно загружены из {MODEL_DIR}.")
except FileNotFoundError:
    print(f"> Ошибка: Файлы модели не найдены в директории {MODEL_DIR}.")
    print("> Проверьте путь к папке.")
    exit()

def correct_word(input_word: str, current_model, current_tokenizer) -> str:
    prefixed_text = 'fix spelling: ' + input_word
    inputs = current_tokenizer(
        prefixed_text,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    model_device = current_model.device
    input_ids = inputs.input_ids.to(model_device)
    attention_mask = inputs.attention_mask.to(model_device)

    with torch.no_grad():
        if XLA_AVAILABLE and model_device.type == 'xla':
            outputs = current_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True,
            ).cpu()
        else:
            outputs = current_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

    return current_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":    
    while True:
        try:
            user_input = input("\n< ")
            if user_input.lower() in ('exit', 'quit'):
                print("> Завершение работы.")
                break
            if not user_input.strip():
                continue
            clean_word = re.sub(r'[^а-яА-ЯёЁa-zA-Z\-]', '', user_input).lower()
            if not clean_word:
                print("> Пожалуйста, введите корректное слово.")
                continue
            result = correct_word(clean_word, model, tokenizer)
            print(f"> Слово с ошибкой: {user_input}")
            print(f"> Исправлено: {result}")
            
        except Exception as e:
            print(f"\n> Произошла ошибка при генерации: {e}")
            break