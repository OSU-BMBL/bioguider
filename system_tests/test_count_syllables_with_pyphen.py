import pyphen
import re
import math
import logging

from bioguider.agents.agent_utils import read_file

logger = logging.getLogger(__name__)

dic = pyphen.Pyphen(lang='en')

def count_syllables(word):
    """Approximate syllable count using hyphenation."""
    return dic.inserted(word).count('-') + 1 if word.isalpha() else 0

def split_sentences(text):
    """Split text into sentences using basic punctuation rules."""
    return re.split(r'[.!?]+', text)

def split_words(text):
    """Split text into words, removing non-alpha tokens."""
    return re.findall(r'\b\w+\b', text)

def is_polysyllabic(word):
    return count_syllables(word) >= 3

def is_complex(word):
    # You can enhance this check to exclude proper nouns, hyphenated words, etc.
    return is_polysyllabic(word)

def readability_metrics(text):
    sentences = [s for s in split_sentences(text) if s.strip()]
    sentence_count = len(sentences)
    
    words = split_words(text)
    word_count = len(words)

    syllable_count = sum(count_syllables(w) for w in words)
    polysyllables = sum(1 for w in words if is_polysyllabic(w))
    complex_words = sum(1 for w in words if is_complex(w))

    # Avoid division by zero
    words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    syllables_per_word = syllable_count / word_count if word_count > 0 else 0
    complex_per_word = complex_words / word_count if word_count > 0 else 0

    # Formulas
    flesch_reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    flesch_kincaid_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    gunning_fog_index = 0.4 * (words_per_sentence + 100 * complex_per_word)
    smog_index = 1.043 * math.sqrt(polysyllables * (30 / sentence_count)) + 3.1291 if sentence_count >= 1 else 0

    

    return sentence_count, word_count, syllable_count, polysyllables, complex_words, flesch_reading_ease, \
    flesch_kincaid_grade, gunning_fog_index, smog_index

def test_biochatter_with_pyphen():
    readme_content = read_file("./data/repos/biochatter/README.md")
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
