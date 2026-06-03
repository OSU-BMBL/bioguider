import os
from datetime import datetime
import logging

import pytest
from dotenv import load_dotenv

from bioguider.agents.agent_utils import get_configured_llm

load_dotenv(override=True)

# ============================================================================================
# utils
@pytest.fixture(scope="session", autouse=True)
def prepare_logging():
    level = logging.INFO
    logging.basicConfig(level=level)
    file_handler = logging.FileHandler("./logs/benchmark.log")
    file_handler.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

def get_litellm():
    # Honor LLM_PROVIDER (kimi / minimax / azure). When unset, falls back to the
    # azure branch, which still passes base_url=OPENAI_BASE_URL — i.e. identical
    # to the previous proxy behavior when OPENAI_BASE_URL is set.
    return get_configured_llm()


@pytest.fixture(scope="module")
def llm():
    return get_litellm()


@pytest.fixture
def test_output_dir():
    """Create output directory for test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs/single_file_stress", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

@pytest.fixture
def test_pipeline_output_dir():
    """Create output directory for test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs/pipeline_stress", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
