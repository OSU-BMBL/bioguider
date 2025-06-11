import logging
import tiktoken
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