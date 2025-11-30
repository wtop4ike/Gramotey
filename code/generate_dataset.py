import os
import re
import random
import pandas as pd
from docx import Document
from tqdm import tqdm
from sklearn.model_selection import train_test_split

keyboard_matrix = {
    'ё': (0, 0), '1': (0, 1), '2': (0, 2), '3': (0, 3), '4': (0, 4), '5': (0, 5), '6': (0, 6), '7': (0, 7), '8': (0, 8), '9': (0, 9), '0': (0, 10), '-': (0, 11), '=': (0, 12),
    'й': (1, 0), 'ц': (1, 1), 'у': (1, 2), 'к': (1, 3), 'е': (1, 4), 'н': (1, 5), 'г': (1, 6), 'ш': (1, 7), 'щ': (1, 8), 'з': (1, 9), 'х': (1, 10), 'ъ': (1, 11), '\\': (1, 12),
    'ф': (2, 0), 'ы': (2, 1), 'в': (2, 2), 'а': (2, 3), 'п': (2, 4), 'р': (2, 5), 'о': (2, 6), 'л': (2, 7), 'д': (2, 8), 'ж': (2, 9), 'э': (2, 10),
    'я': (3, 0), 'ч': (3, 1), 'с': (3, 2), 'м': (3, 3), 'и': (3, 4), 'т': (3, 5), 'ь': (3, 6), 'б': (3, 7), 'ю': (3, 8), '.': (3, 9), ',': (3, 10)
}
coordinate_to_key = {coord: key for key, coord in keyboard_matrix.items()}

en_to_ru = {
    'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г', 'i': 'ш', 'o': 'щ', 'p': 'з', '[': 'х', ']': 'ъ', 'a': 'ф', 's': 'ы', 'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о', 'k': 'л', 'l': 'д', ';': 'ж', "'": 'э', 'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и', 'n': 'т', 'm': 'ь', ',': 'б', '.': 'ю', '/': '.',
    'Q': 'Й', 'W': 'Ц', 'E': 'У', 'R': 'К', 'T': 'Е', 'Y': 'Н', 'U': 'Г', 'I': 'Ш', 'O': 'Щ', 'P': 'З', '{': 'Х', '}': 'Ъ', 'A': 'Ф', 'S': 'Ы', 'D': 'В', 'G': 'П', 'H': 'Р', 'J': 'О', 'K': 'Л', 'L': 'Д', ':': 'Ж', '"': 'Э', 'Z': 'Я', 'X': 'Ч', 'C': 'С', 'V': 'М', 'B': 'И', 'N': 'Т', 'M': 'Ь', '<': 'Б', '>': 'Ю', '?': ','
}
ru_to_en = {v: k for k, v in en_to_ru.items()}

def get_neighboring_keys(char):
    if char.lower() not in keyboard_matrix: return []
    row, col = keyboard_matrix[char.lower()]
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            new_coord = (row + dr, col + dc)
            if new_coord in coordinate_to_key:
                neighbor_char = coordinate_to_key[new_coord]
                if char.isupper(): neighbor_char = neighbor_char.upper()
                neighbors.append(neighbor_char)
    return neighbors

def introduce_keyboard_typos(word, typo_prob=0.2):
    if len(word) < 3: return word
    chars = list(word)
    if random.random() < typo_prob:
        pos = random.randint(1, len(word)-2)
        char = chars[pos]; neighbors = get_neighboring_keys(char)
        if neighbors: chars[pos] = random.choice(neighbors)
    return ''.join(chars)

def introduce_layout_typos(text, typo_prob=1.0):
    if random.random() > typo_prob: return text
    new_word = []
    for char in text: new_word.append(ru_to_en.get(char, en_to_ru.get(char, char)))
    return ''.join(new_word)

def introduce_other_typos(word, typo_prob=0.15):
    if random.random() > typo_prob or len(word) < 4: return word
    typo_type = random.choice(['missing', 'extra', 'swap'])
    if typo_type == 'missing':  
        pos = random.randint(1, len(word)-2); return word[:pos] + word[pos+1:]
    elif typo_type == 'extra':  
        pos = random.randint(1, len(word)-1); extra_char = random.choice('абвгдежзийклмнопрстуфхцчшщъыьэюя')
        return word[:pos] + extra_char + word[pos:]
    elif typo_type == 'swap':  
        pos = random.randint(0, len(word)-2); return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

def extract_text_from_docx(file_path):
    """Извлекает текст из Word документа."""
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip() != '']
        return '\n'.join(paragraphs)
    except Exception as e:
        print(f"Ошибка при извлечении текста из {file_path}: {e}")
        return ""

def is_valid_word(word, min_len=3):
    word_lower = word.lower()
    if len(word_lower) < min_len: return False
    if word.replace('-', '').replace('.', '', 1).isdigit(): return False
    if not re.fullmatch(r'[\w\-]+', word_lower): return False
    cyrillic_count = len(re.findall(r'[а-яё]', word_lower))
    alphabetic_count = len(re.findall(r'[а-яёa-z]', word_lower))
    if alphabetic_count == 0 or (cyrillic_count / alphabetic_count) < 0.8: return False
    
    return True

def generate_word_typos(original_word):
    """Генерирует один вариант ошибки для одного слова."""
    typo_type = random.choice(['layout', 'keyboard', 'other'])
    
    if typo_type == 'layout': 
        corrupted_word = introduce_layout_typos(original_word, 1.0)
    elif typo_type == 'keyboard':
        corrupted_word = introduce_keyboard_typos(original_word, 0.9)
    else:
        corrupted_word = introduce_other_typos(original_word, 0.9)
            
    return corrupted_word


def generate_word_dataset(file_path, output_dir="./", error_rate=0.8):
    """
    Генерирует пары СЛОВО-СЛОВО (Input, Target) и сохраняет финальные CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_text = extract_text_from_docx(file_path)
    all_words = re.findall(r'[а-яА-ЯёЁ\w]+', raw_text)
    data = []
    unique_words = set()
    for word in all_words:
        if is_valid_word(word):
             unique_words.add(word.lower())

    for word in tqdm(unique_words, desc="Обработка слов"):
        target = word
        
        if random.random() < error_rate:
            corrupted = generate_word_typos(word)
            
            if corrupted != target:
                data.append({
                    'input_text': 'fix spelling: ' + corrupted,
                    'target_text': target
                })
        else:
            data.append({
                'input_text': 'fix spelling: ' + target,
                'target_text': target
            })
            
    df_words = pd.DataFrame(data).drop_duplicates()

    train_df, temp_df = train_test_split(df_words, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df.to_csv(os.path.join(output_dir, "train_words.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_words.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_words.csv"), index=False)
    
    print(f"Train (слов): {len(train_df)}, Val (слов): {len(val_df)}, Test (слов): {len(test_df)}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    generate_word_dataset(
        file_path="./input.docx", 
        output_dir="./",
        error_rate=0.8
    )