"""
    This file is part of ML_mini_proj project
    Copyright (C) 2024 Yao Shu  <springrainyszxr@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import re
import string
import jieba
import nltk
import contractions
from nltk.tokenize import word_tokenize

# Download necessary nltk resources (only required for first-time execution)
nltk.download('punkt')

# List of filler words
ENGLISH_FILLERS = ['um', 'uh', 'you know']
CHINESE_FILLERS = ['嗯', '啊', '这个']

LANGUAGE_EN = "English"
LANGUAGE_ZH = "Chinese"


def remove_fillers(text, language):
    """
    Remove filler and hesitation words, choosing the filler list based on the language
    """
    fillers = ENGLISH_FILLERS if language == 'en' else CHINESE_FILLERS
    for filler in fillers:
        pattern = r'\b{}\b'.format(re.escape(filler))
        text = re.sub(pattern, '', text)
    return text


def remove_non_verbal(text):
    """
    Remove non-verbal sound annotations such as [laughs], [sighs]
    """
    pattern = r'\[.*?\]'
    text = re.sub(pattern, '', text)
    return text


def lowercase_text(text, language):
    """
    Convert text to lowercase for English only
    """
    if language == LANGUAGE_EN:
        return text.lower()
    return text


def remove_punctuation(text):
    """
    Remove English punctuation
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    # If handling Chinese punctuation is needed, additional logic can be added here
    return text


def expand_contractions_func(text, language):
    """
    Expand English contractions, only for English
    """
    if language == LANGUAGE_EN:
        return contractions.fix(text)
    return text


def normalize_whitespace(text):
    """
    Normalize whitespace characters
    """
    return ' '.join(text.split())


def tokenize_english(text):
    """
    Tokenize English text
    """
    return word_tokenize(text)


def tokenize_chinese(text):
    """
    Tokenize Chinese text
    """
    return list(jieba.cut(text))


def mixed_tokenize(text, language):
    """
    Tokenize text using the appropriate method based on the language
    """
    if language == LANGUAGE_EN:
        return tokenize_english(text)
    elif language == LANGUAGE_ZH:
        return tokenize_chinese(text)
    else:
        # Default to English tokenization
        return tokenize_english(text)


def preprocess_text(text, language):
    """
    Complete text preprocessing pipeline
    """
    text = remove_fillers(text, language)
    text = remove_non_verbal(text)
    text = lowercase_text(text, language)
    text = expand_contractions_func(text, language)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    tokens = mixed_tokenize(text, language)
    return tokens
