
import pytest
import logging
from textstat import textstat

from langchain_core.prompts import ChatPromptTemplate
from bioguider.agents.agent_utils import read_file
from bioguider.agents.common_agent import CommonConversation

logger = logging.getLogger(__name__)

COUNT_SYLLABLES_SYSTEM_PROMPT = """
You are an expert in linguistic analysis. Your task is to **count syllables in English words accurately** to support readability scoring using formulas like Flesch Reading Ease, Flesch-Kincaid Grade Level, and SMOG Index.

### **Syllable Counting Rules:**

Follow these rules precisely for each word:

1. **Each vowel group (a, e, i, o, u, y)** that sounds like one unit of speech counts as **one syllable**. For example, “education” has 4 syllables: e-du-ca-tion.
2. Do **not** count silent vowels (e.g., the final 'e' in "make" is silent).
3. Two adjacent vowels that form a single sound (a diphthong like "ai", "ou", "ea") count as **one syllable** (e.g., "bread" = 1 syllable).
4. Words ending in “-le” preceded by a consonant typically count as an extra syllable (e.g., “table” = 2 syllables).
5. If a word has **prefixes or suffixes**, count them only if they contain a separate vowel sound (e.g., “unwanted” = un-want-ed = 3 syllables).

### **Examples:**

* “make” → 1 syllable
* “bread” → 1 syllable
* “education” → 4 syllables
* “syllable” → 3 syllables
* “queue” → 1 syllable
* “really” → 2 syllables

### **Your Task:**

Given the following text, return:

1. Total number of **words**
2. Total number of **sentences**
3. Total number of **syllables** (following the rules above)

**Text:**

{input_text}

"""

@pytest.mark.skip()
def test_count_syllables_POPPER(llm):
    readme_content = read_file("./data/repos/POPPER/README.md")
    readme_content = readme_content.replace("{", "{{").replace("}", "}}")
    system_prompt = ChatPromptTemplate.from_template(
        COUNT_SYLLABLES_SYSTEM_PROMPT
    ).format(input_text=readme_content)

    conversation = CommonConversation(llm=llm)
    res, _ = conversation.generate(
        system_prompt=system_prompt,
        instruction_prompt="Now, let's count."
    )
    assert res is not None
    logger.info("="*84)
    logger.info("POPPER")
    logger.info(res)

@pytest.mark.skip()
def test_count_syllables_RepoAgent(llm):
    readme_content = read_file("./data/repos/RepoAgent/README.md")
    readme_content = readme_content.replace("{", "{{").replace("}", "}}")
    system_prompt = ChatPromptTemplate.from_template(
        COUNT_SYLLABLES_SYSTEM_PROMPT
    ).format(input_text=readme_content)

    conversation = CommonConversation(llm=llm)
    res, _ = conversation.generate(
        system_prompt=system_prompt,
        instruction_prompt="Now, let's count."
    )
    assert res is not None
    logger.info("="*84)
    logger.info("RepoAgent")
    logger.info(res)

def test_textstat_RepoAgent(llm):
    readme_content = read_file("./data/repos/RepoAgent/README.md")
    text = readme_content
    syllable_count = textstat.syllable_count(text)
    sentence_count = textstat.sentence_count(text)
    polysyllable_count = textstat.polysyllabcount(text)
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    logger.info("="*84)
    logger.info("RepoAgent textstat")
    logger.info(f"syllable count: {syllable_count}")
    logger.info(f"sentence count: {sentence_count}")
    logger.info(f"polysyllable count: {polysyllable_count}")
    logger.info(f"flesch reading ease: {flesch_reading_ease}")
    logger.info(f"flesch kincaid grade: {flesch_kincaid_grade}")
    logger.info(f"gunning fog: {gunning_fog}")
    logger.info(f"smog index: {smog_index}")

def test_textstat_biochatter(llm):
    readme_content = read_file("./data/repos/biochatter/README.md")
    text = readme_content
    syllable_count = textstat.syllable_count(text)
    sentence_count = textstat.sentence_count(text)
    polysyllable_count = textstat.polysyllabcount(text)
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    logger.info("="*84)
    logger.info("biochatter textstat")
    logger.info(f"syllable count: {syllable_count}")
    logger.info(f"sentence count: {sentence_count}")
    logger.info(f"polysyllable count: {polysyllable_count}")
    logger.info(f"flesch reading ease: {flesch_reading_ease}")
    logger.info(f"flesch kincaid grade: {flesch_kincaid_grade}")
    logger.info(f"gunning fog: {gunning_fog}")
    logger.info(f"smog index: {smog_index}")




