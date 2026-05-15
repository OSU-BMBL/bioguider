import os
from datetime import datetime
import logging

import pytest
from dotenv import load_dotenv

from bioguider.agents.agent_utils import get_llm

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
    return get_llm(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        model_name=os.environ.get("OPENAI_MODEL", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_DEPLOYMENT_NAME", None),
        max_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKEN", 16384)),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
    )


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
