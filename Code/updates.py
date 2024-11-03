import csv
import requests
import re
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from dotenv import load_dotenv
import os

# Instantiate the smoothing function
chencherry = SmoothingFunction()

# Load environment variables from .env file
load_dotenv()

# Access environment variables
translator_key = os.getenv("TRANSLATOR_KEY")
service_region = os.getenv("SERVICE_REGION")
translator_endpoint = os.getenv("TRANSLATOR_ENDPOINT")

# Default target language for translation
target_language = "kn"  # Kannada
source_language = "en-GB"  # English (Great Britain)

# Step 1: Load the Kannada-Tulu sentence pairs from 'kn.txt' and 'tulu.txt'
def load_sentence_pairs(kannada_file, tulu_file):
    with open(kannada_file, 'r', encoding='utf-8') as kn, open(tulu_file, 'r', encoding='utf-8') as tu:
        kannada_sentences = kn.readlines()
        tulu_sentences = tu.readlines()
    
    # Remove extra whitespace and ensure both lists have the same number of lines
    kannada_sentences = [line.strip() for line in kannada_sentences]
    tulu_sentences = [line.strip() for line in tulu_sentences]
    
    return kannada_sentences, tulu_sentences

# Step 2: Create a word/phrase dictionary from the sentence pairs
def build_translation_dict(kannada_sentences, tulu_sentences):
    translation_dict = {}
    for kn_sentence, tu_sentence in zip(kannada_sentences, tulu_sentences):
        kn_words = kn_sentence.split()
        tu_words = tu_sentence.split()
        for kn_word, tu_word in zip(kn_words, tu_words):
            if kn_word not in translation_dict:
                translation_dict[kn_word] = tu_word
    return translation_dict

# Function to translate text from English (GB) to Kannada using Azure Translator API
def translate_to_kannada(text):
    path = '/translate?api-version=3.0'
    params = f'&from={source_language}&to={target_language}'
    constructed_url = translator_endpoint + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': translator_key,
        'Ocp-Apim-Subscription-Region': service_region,
        'Content-type': 'application/json',
    }

    body = [{'text': text}]
    response = requests.post(constructed_url, headers=headers, json=body)

    if response.status_code == 200:
        result = response.json()
        translated_text = result[0]['translations'][0]['text']
        return translated_text
    else:
        return f"Error: {response.status_code}, {response.text}"

# Step 3: Translate a Kannada sentence to Tulu using the dictionary
def translate_sentence_to_tulu(kannada_sentence, translation_dict):
    words = kannada_sentence.split()
    translated_sentence = []
    for word in words:
        translated_word = translation_dict.get(word, word)
        translated_sentence.append(translated_word)
    return ' '.join(translated_sentence)

# Function to save translations to a CSV file
def save_translation_to_csv(original_text, kannada_text, tulu_text, filename='translated_text.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Original Text', 'Kannada Translation', 'Tulu Translation'])
        writer.writerow([original_text, kannada_text, tulu_text])

# Load Kannada-Tulu sentence pairs and create the dictionary
kannada_sentences, tulu_sentences = load_sentence_pairs(r'Dataset\Training\kn_tr.txt', r'Dataset\Training\tu_tr.txt')
translation_dict = build_translation_dict(kannada_sentences, tulu_sentences)

# Example usage
text_to_translate = input("Enter English text to translate: ")
kannada_text = translate_to_kannada(text_to_translate)
print(f'Kannada Translation: {kannada_text}')
tulu_text = translate_sentence_to_tulu(kannada_text, translation_dict)
print(f'Tulu Translation: {tulu_text}')
save_translation_to_csv(text_to_translate, kannada_text, tulu_text)
print("Translation saved to CSV file.")

# Step 4: Load the validation data
def load_validation_data(kannada_file, tulu_file):
    with open(kannada_file, 'r', encoding='utf-8') as kn, open(tulu_file, 'r', encoding='utf-8') as tu:
        kannada_sentences = [line.strip() for line in kn.readlines()]
        tulu_sentences = [line.strip() for line in tu.readlines()]
    return kannada_sentences, tulu_sentences

# Step 5: Calculate BLEU score for Kannada-Tulu translations
def calculate_bleu_score(references, hypotheses):
    # Use smoothing in sentence-level BLEU
    bleu_scores = [
        sentence_bleu([ref.split()], hyp.split(), smoothing_function=chencherry.method1)
        for ref, hyp in zip(references, hypotheses)
    ]
    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    # Use smoothing in corpus-level BLEU
    corpus_bleu_score = corpus_bleu(
        [[ref.split()] for ref in references],
        [hyp.split() for hyp in hypotheses],
        smoothing_function=chencherry.method1
    )
    return average_bleu, corpus_bleu_score

# Step 6: Calculate accuracy (word-level or exact sentence match)
def calculate_accuracy(references, hypotheses):
    exact_matches = sum(1 for ref, hyp in zip(references, hypotheses) if ref == hyp)
    accuracy = (exact_matches / len(references)) * 100 if references else 0  # Convert to percentage
    return accuracy

# Step 7: Evaluate translations
def evaluate_translations(kannada_sentences, tulu_sentences, translation_dict):
    translated_sentences = [translate_sentence_to_tulu(sentence, translation_dict) for sentence in kannada_sentences]
    bleu_score, corpus_bleu_score = calculate_bleu_score(tulu_sentences, translated_sentences)
    accuracy = calculate_accuracy(tulu_sentences, translated_sentences)
    return bleu_score, corpus_bleu_score, accuracy

# Load validation data
kannada_val_sentences, tulu_val_sentences = load_validation_data(
    r'Dataset\Validation\kn_dev.txt', r'Dataset\Validation\tu_dev.txt'
)

# Evaluate translations
bleu_score, corpus_bleu_score, accuracy = evaluate_translations(kannada_val_sentences, tulu_val_sentences, translation_dict)

print(f"Sentence-level BLEU Score: {bleu_score:.4f}")
print(f"Corpus-level BLEU Score: {corpus_bleu_score:.4f}")
print(f"Accuracy Score: {accuracy:.4f}")
