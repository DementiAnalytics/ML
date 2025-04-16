import os
import nltk
import spacy
import syllapy

# Importing tokenizers
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


def process_and_rename(data_dir):
    """
    Preprocesses files to include 'healthy' and 'dementia' markers in the filenames for further labeling in the model
    """
    for label_folder in ['control', 'dementia']:
        folder_path = os.path.join(data_dir, label_folder)
        label_prefix = "healthy" if label_folder == "control" else "dementia"

        for idx, filename in enumerate(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                old_path = os.path.join(folder_path, filename)
                new_filename = f"{label_prefix}_{idx}.txt"
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)

def extract_features(text):
    """
    Extracts several linguistic features based on the provided text
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    words = text.split()
    unique_words = set(words)

    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0

    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    pronoun_count = sum(1 for token in doc if token.pos_ == "PRON")
    adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")

    noun_rate = noun_count / word_count if word_count > 0 else 0
    pronoun_rate = pronoun_count / word_count if word_count > 0 else 0
    adjective_rate = adjective_count / word_count if word_count > 0 else 0
    verb_rate = verb_count / word_count if word_count > 0 else 0

    filler_words = ['uh', 'um', 'like', 'you know']
    filler_count = sum(text.lower().split().count(filler) for filler in filler_words)

    syllable_count = sum(syllapy.count(word) for word in words)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": type_token_ratio,
        "noun_rate": noun_rate,
        "pronoun_rate": pronoun_rate,
        "adjective_rate": adjective_rate,
        "verb_rate": verb_rate,
        "filler_count": filler_count,
        "syllable_count": syllable_count
    }

def create_labels(data_dir):
    """
    Creates data labels and feature list for visualization and model input
    """
    features_list = []
    labels = []

    for label_folder in ["control", "dementia"]:
        folder_path = os.path.join(data_dir, label_folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                features = extract_features(text)
                features_list.append(features)
                
                if filename.lower().startswith("healthy"):
                    labels.append(0)
                elif filename.lower().startswith("dementia"):
                    labels.append(1)
    
    return features_list, labels