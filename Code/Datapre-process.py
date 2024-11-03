import os
import re

# File paths for training and validation data
train_files = {
    'kn': r'Dataset/Training/kn_tr.txt',
    'tu': r'Dataset/Training/tu_tr.txt',
    'en': r'Dataset/Training/en_tr.txt'
}

dev_files = {
    'kn': r'Dataset/Validation/kn_dev.txt',
    'tu': r'Dataset/Validation/tu_dev.txt',
    'en': r'Dataset/Validation/en_dev.txt'
}

# Directory to save processed files
output_dir = 'processed_dataset'
os.makedirs(output_dir, exist_ok=True)

# Function to clean text by removing special characters, lowercasing, etc.
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to preprocess a single file
def preprocess_file(file_path):
    cleaned_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = clean_text(line)
            cleaned_lines.append(cleaned_line)
    return cleaned_lines

# Function to process and save cleaned data to the processed dataset folder
def preprocess_and_save(files, output_suffix='_cleaned'):
    processed_data = {}
    for lang, file_path in files.items():
        processed_data[lang] = preprocess_file(file_path)
        output_file = os.path.join(output_dir, f"{lang}{output_suffix}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in processed_data[lang]:
                f.write(line + '\n')
        print(f"Processed and saved {output_file}")
    return processed_data

# Preprocess training and validation files
train_data = preprocess_and_save(train_files, '_cleaned')
dev_data = preprocess_and_save(dev_files, '_cleaned')
