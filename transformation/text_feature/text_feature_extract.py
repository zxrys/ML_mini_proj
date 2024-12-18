import os

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import spacy
import pkuseg
from tqdm import tqdm

import utils
from transformation.text_feature.text_preprocess import preprocess_text

# Initialize the multilingual BERT model
mbert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
mbert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Initialize spaCy English model
try:
    nlp_en = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    spacy.cli.download('en_core_web_sm')
    nlp_en = spacy.load('en_core_web_sm')

# Initialize Chinese tokenizer (load once)
seg = pkuseg.pkuseg()


def get_mbert_embeddings(text):
    """
    Get [CLS] embeddings from multilingual BERT
    """
    inputs = mbert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = mbert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding.numpy()


def get_mean_pooling_embeddings(text):
    """
    Get mean pooling embeddings from multilingual BERT
    """
    inputs = mbert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = mbert_model(**inputs)
    attention_mask = inputs['attention_mask']
    embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    summed = torch.sum(embeddings * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / counts
    return mean_pooled.numpy()


def compute_syntactic_complexity_en(text):
    """
    Compute syntactic complexity for English text
    """
    doc = nlp_en(text)
    sentences = list(doc.sents)
    if not sentences:
        return (0, 0, 0)
    avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
    avg_word_length = sum(len(token) for token in doc) / len(doc) if doc else 0
    # clause_density = sum(1 for token in doc if token.dep_ == 'csubj') / len(sentences) if sentences else 0
    return (avg_sentence_length, avg_word_length)


def compute_syntactic_complexity_zh(text):
    """
    Compute syntactic complexity for Chinese text
    """
    words = seg.cut(text)
    if text.count('。') == 0:
        avg_sentence_length = len(words)
    else:
        avg_sentence_length = len(words) / text.count('。')
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    return (avg_sentence_length, avg_word_length)


def perform_topic_modeling(texts, language='en', num_topics=5):
    """
    Perform topic modeling (optional)
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    if language == 'zh':
        vectorizer = CountVectorizer(tokenizer=lambda x: preprocess_text(x, 'zh'))
    else:
        vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    return lda


def aggregate_features(embedding, syntactic_complexity):
    """
    Aggregate all features into one vector
    """
    embedding = np.array(embedding)
    syntactic_complexity = np.array(syntactic_complexity)
    assert embedding.ndim == 1, f"Embedding shape is incorrect: {embedding.shape}"
    assert syntactic_complexity.ndim == 1, f"Syntactic complexity shape is incorrect: {syntactic_complexity.shape}"

    return np.concatenate([embedding, syntactic_complexity])


def extract_text_feature(text, language):
    """
    Extract all features from a single text
    """
    # Preprocessing
    tokens = preprocess_text(text, language)
    cleaned_text = ' '.join(tokens)

    # BERT embeddings
    mbert_embedding = get_mbert_embeddings(cleaned_text)

    # Syntactic complexity
    if language == 'English':
        syntactic_features = compute_syntactic_complexity_en(cleaned_text)
    elif language == 'Chinese':
        syntactic_features = compute_syntactic_complexity_zh(cleaned_text)
    else:
        syntactic_features = (0, 0, 0)  # Default value or other handling

    # Aggregate features
    aggregated_features = aggregate_features(mbert_embedding, syntactic_features)

    return aggregated_features


def load_labels_with_language(csv_path):
    """
    Load labels from a CSV file and include Language in the mapping.
    The CSV should have columns: filename, Language, Story_type
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Map Story_type to labels
    label_mapping = {'True Story': 0, 'Deceptive Story': 1}
    df['label'] = df['Story_type'].map(label_mapping)

    # Check for missing or invalid Story_type values
    if df['label'].isnull().any():
        missing = df[df['label'].isnull()]
        raise ValueError(f"Some Story_type values are invalid or missing: {missing}")

    # Create a dictionary mapping filename to a dict containing label and Language
    filename_to_label = df.set_index('filename')[['label', 'Language']].to_dict(orient='index')

    return filename_to_label


def get_text_file_paths(text_directory, extension='.txt'):
    """Retrieve all text file paths with the given extension."""
    return [
        os.path.join(text_directory, file)
        for file in os.listdir(text_directory)
        if file.endswith(extension)
    ]


def load_text_files(text_files):
    """Load text files from the given file paths."""
    texts = []
    for file_path, language in text_files:
        with open(file_path, "r") as f:
            text = f.read()
        texts.append((text, language))
    return texts


def extract_text_features(text_files):
    feature_matrix = []
    for text, language in tqdm(text_files, "Extract features"):
        feature_matrix.append(extract_text_feature(text, language))
    feature_matrix = np.array(feature_matrix)
    return feature_matrix


def process_and_save_features(csv_file, text_paths, feature_path, label_path):
    """
    Process text files and save extracted features and labels.

    Args:
        csv_file (str): Path to the CSV file containing labels and metadata.
                        The CSV file should have columns: filename, Language, Story_type.
        text_paths (list[str]): Path to the root directory containing text files.
        feature_path (str): Path to save the extracted feature matrix (numpy format).
        label_path (str): Path to save the corresponding labels (numpy format).

    Steps:
        1. Load labels from the CSV file.
        2. Match text files with their corresponding metadata using filenames.
        3. Load and preprocess text files.
        4. Extract features (e.g., BERT embeddings, syntactic complexity) from each text.
        5. Aggregate features into a matrix.
        6. Extract and save labels corresponding to the text files.
        7. Save features and labels to specified file paths.
    """
    # Step 1: Load labels from the CSV file
    filename_to_label = load_labels_with_language(csv_file)
    print(f"Loaded labels for {len(filename_to_label)} files from {csv_file}.")

    # Step 2: Match text files with their metadata (language and labels)
    text_files = []
    for file_path in text_paths:
        wav_name = os.path.basename(file_path).replace("txt", "wav")  # Convert txt to wav filename
        if wav_name in filename_to_label:
            language = filename_to_label.get(wav_name)['Language']
            text_files.append((file_path, language))  # Add file path and language to process
    text_files = load_text_files(text_files)  # Load the content of the text files
    print(f"Got {len(text_files)} data to process")

    # Step 3 & 4: Extract features for each text file
    feature_matrix = extract_text_features(text_files)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Step 5: Extract labels corresponding to the text files
    labels = []
    for file_path in text_paths:
        filename = os.path.basename(file_path).replace("txt", "wav")  # Convert txt to wav filename
        label = filename_to_label.get(filename)['label']  # Get label for the file
        if label is not None:
            labels.append(label)
        else:
            print(f"Warning: No label found for {filename}, skipping.")  # Warn if no label is found
    labels = np.array(labels)  # Convert labels to a NumPy array
    print(f"Labels shape: {labels.shape}")

    # Step 6: Save the feature matrix and labels to the specified paths
    utils.save_features(feature_matrix, labels, feature_path, label_path)
