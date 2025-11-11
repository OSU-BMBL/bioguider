import logging
from pathlib import Path
import re
import subprocess
from typing import Optional
from pydantic import BaseModel
import tiktoken
from bs4 import BeautifulSoup

from bioguider.utils.constants import DEFAULT_TOKEN_USAGE
logger = logging.getLogger(__name__)

def count_tokens(text: str, local_ollama: bool = False) -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text (str): The text to count tokens for.
        local_ollama (bool, optional): Whether using local Ollama embeddings. Default is False.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        if local_ollama:
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        # Rough approximation: 4 characters per token
        return len(text) // 4


def run_command(command: list, cwd: str = None, timeout: int = None):
    """
    Run a shell command with optional timeout and return stdout, stderr, and return code.
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        return e.stdout or "", e.stderr or f"Command timed out after {timeout} seconds", -1

def escape_braces(text: str) -> str:
    def fix_braces(m):
        s = m.group(0)
        # If odd number of braces, double the last one
        if len(s) % 2 == 1:
            return s + s[-1]
        return s
    # Handle both { and } sequences
    text = re.sub(r'{+|}+', fix_braces, text)
    return text

def increase_token_usage(
    token_usage: Optional[dict] = None,
    incremental: dict = {**DEFAULT_TOKEN_USAGE},
):
    if token_usage is None:
        token_usage = {**DEFAULT_TOKEN_USAGE}
    token_usage["total_tokens"] += incremental["total_tokens"]
    token_usage["completion_tokens"] += incremental["completion_tokens"]
    token_usage["prompt_tokens"] += incremental["prompt_tokens"]

    return token_usage

def clean_action_input(action_input: str) -> str:
    replaced_input = ""

    while (True):
        replaced_input = action_input.strip()
        replaced_input = replaced_input.strip("`")
        replaced_input = replaced_input.strip('"')
        replaced_input = replaced_input.strip()
        replaced_input = replaced_input.strip("`")
        replaced_input = replaced_input.strip('"')
        replaced_input = replaced_input.strip()
        if (replaced_input == action_input):
            break
        action_input = replaced_input
    
    action_input = action_input.replace("'", '"')
    action_input = action_input.replace("`", '"')
    return action_input

# Convert BaseModel objects to dictionaries for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def convert_html_to_text(html_path: str | Path, exclude_tags: list[str] = ["script", "style", "img", "svg", "meta", "link"]) -> str:
    """
    This function is used to convert html string to text, that is,
    extract text from html content, including tables.
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"File {html_path} does not exist")
    with html_path.open("r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    if exclude_tags is not None:
        for tag in exclude_tags:
            for element in soup.find_all(tag):
                element.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text

def get_overall_score(grade_levels: list[int | bool], weights: list[int]) -> int:
    max_score = 100
    min_score = 0
    def get_grade_level_score(grade_level: int | bool) -> int:
        if isinstance(grade_level, bool):
            if grade_level:
                return max_score
            else:
                return min_score
        return grade_level
    if len(grade_levels) != len(weights):
        raise ValueError("The length of grade_levels and weights must be the same")
    score = round(sum(
        get_grade_level_score(grade_level) * weight for grade_level, weight in zip(grade_levels, weights)
    ) / sum(weight * max_score for weight in weights), 2)
        
    return int(score*max_score) if score > 0 else min_score

    


