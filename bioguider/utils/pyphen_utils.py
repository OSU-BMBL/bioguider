
import os
import re
import pyphen
import math

class PyphenReadability:
    def __init__(self, lang='en'):
        self.dic = pyphen.Pyphen(lang=lang)

    def count_syllables(self, word):
        return self.dic.inserted(word).count('-') + 1 if word.isalpha() else 0
    
    def extract_urls(self, text):
        """Find all URLs in the text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.findall(url_pattern, text)
    
    def remove_urls(self, text):
        """Remove URLs from text for clean sentence splitting."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def split_sentences(self, text):
        """Split into sentences using punctuation."""
        return re.split(r'[.!?]+', text)
    
    def split_words(self, text):
        """Extract words."""
        return re.findall(r'\b\w+\b', text)
    
    def is_polysyllabic(self, word):
        return self.count_syllables(word) >= 3
    
    def is_complex(self, word):
        return self.is_polysyllabic(word)
    
    def readability_metrics(self, text):
        # Extract and remove URLs
        urls = self.extract_urls(text)
        url_count = len(urls)
        text_without_urls = self.remove_urls(text)
    
        # Split and count
        sentences = [s for s in self.split_sentences(text_without_urls) if s.strip()]
        sentence_count = len(sentences) + url_count
    
        words = self.split_words(text) # split_words(text_without_urls)
        word_count = len(words)
    
        syllable_count = sum(self.count_syllables(w) for w in words)
        polysyllables = sum(1 for w in words if self.is_polysyllabic(w))
        complex_words = sum(1 for w in words if self.is_complex(w))
    
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
    
        return flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index,\
             sentence_count, word_count, syllable_count, polysyllables, complex_words
    

    
