import pyphen
import re
import math
import logging

from bioguider.agents.agent_utils import read_file

logger = logging.getLogger(__name__)

dic = pyphen.Pyphen(lang='en')

def count_syllables(word):
    return dic.inserted(word).count('-') + 1 if word.isalpha() else 0

def extract_urls(text):
    """Find all URLs in the text."""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.findall(url_pattern, text)

def remove_urls(text):
    """Remove URLs from text for clean sentence splitting."""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

def split_sentences(text):
    """Split into sentences using punctuation."""
    return re.split(r'[.!?]+', text)

def split_words(text):
    """Extract words."""
    return re.findall(r'\b\w+\b', text)

def is_polysyllabic(word):
    return count_syllables(word) >= 3

def is_complex(word):
    return is_polysyllabic(word)

def readability_metrics(text):
    # Extract and remove URLs
    urls = extract_urls(text)
    url_count = len(urls)
    text_without_urls = remove_urls(text)

    # Split and count
    sentences = [s for s in split_sentences(text_without_urls) if s.strip()]
    sentence_count = len(sentences) + url_count

    words = split_words(text) # split_words(text_without_urls)
    word_count = len(words)

    syllable_count = sum(count_syllables(w) for w in words)
    polysyllables = sum(1 for w in words if is_polysyllabic(w))
    complex_words = sum(1 for w in words if is_complex(w))

    # Avoid division by zero
    words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    syllables_per_word = syllable_count / word_count if word_count > 0 else 0
    complex_per_word = complex_words / word_count if word_count > 0 else 0

    # Readability formulas
    flesch_reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    flesch_kincaid_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    gunning_fog_index = 0.4 * (words_per_sentence + 100 * complex_per_word)
    smog_index = (
        1.043 * math.sqrt(polysyllables * (30 / sentence_count)) + 3.1291
        if sentence_count >= 1 else 0
    )

    return sentence_count, word_count, syllable_count, polysyllables, complex_words, flesch_reading_ease, \
    flesch_kincaid_grade, gunning_fog_index, smog_index

def test_biochatter_with_pyphen():
    readme_content = read_file("./system_tests/test_data/biochatter_README.txt")
    text = readme_content
    sentence_count, word_count, syllable_count, polysyllables, complex_words, flesch_reading_ease, \
         flesch_kincaid_grade, gunning_fog_index, smog_index = readability_metrics(text)
        
    logger.info("="*84)
    logger.info("biochatter pyphen")
    logger.info(f"sentence count: {sentence_count}")
    logger.info(f"word count: {word_count}")
    logger.info(f"syllable count: {syllable_count}")
    logger.info(f"polysyllable count: {polysyllables}")
    logger.info(f"complex word count: {complex_words}")
    logger.info(f"Flesch Reading Ease: {round(flesch_reading_ease, 2)}")
    logger.info(f"Flesch-Kincaid Grade Level: {round(flesch_kincaid_grade, 2)}")
    logger.info(f"Gunning Fog Index: {round(gunning_fog_index, 2)}")
    logger.info(f"SMOG Index: {round(smog_index, 2)}")

def test_RepoAgent_with_pyphen():
    readme_content = read_file("./system_tests/test_data/RepoAgent_README.txt")
    text = readme_content
    sentence_count, word_count, syllable_count, polysyllables, complex_words, flesch_reading_ease, \
         flesch_kincaid_grade, gunning_fog_index, smog_index = readability_metrics(text)
        
    logger.info("="*84)
    logger.info("RepoAgent pyphen")
    logger.info(f"sentence count: {sentence_count}")
    logger.info(f"word count: {word_count}")
    logger.info(f"syllable count: {syllable_count}")
    logger.info(f"polysyllable count: {polysyllables}")
    logger.info(f"complex word count: {complex_words}")
    logger.info(f"Flesch Reading Ease: {round(flesch_reading_ease, 2)}")
    logger.info(f"Flesch-Kincaid Grade Level: {round(flesch_kincaid_grade, 2)}")
    logger.info(f"Gunning Fog Index: {round(gunning_fog_index, 2)}")
    logger.info(f"SMOG Index: {round(smog_index, 2)}")
