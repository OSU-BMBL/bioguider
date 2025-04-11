import os
from typing import Optional
from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import pytest
from dotenv import load_dotenv
import logging

from bioguider.agents.agent_utils import DEFAULT_TOKEN_USAGE, increase_token_usage



def get_openai():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL_NAME"),
    )


def get_azure_openai():
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_4O_MODEL", None),
        max_retries=5,
        # temperature=0.0,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
    )


def get_deepseek():
    return ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        temperature=0.0,
        max_completion_tokens=10000,
        max_retries=3,
    )

@pytest.fixture(scope="module")
def llm():
    return get_azure_openai()

@pytest.fixture(scope="module")
def step_callback():
    total_tokens = {**DEFAULT_TOKEN_USAGE}

    def print_step(
        step_name: Optional[str] = None,
        step_description: Optional[str] = None,
        step_output: Optional[str] = None,
        step_reasoning_process: Optional[str] = None,
        token_usage: Optional[dict] = None,
    ):
        nonlocal total_tokens
        logger = logging.getLogger(__name__)
        if step_name is not None:
            logger.info("=" * 64)
            logger.info(step_name)
        if step_description is not None:
            logger.info(step_description)
        if token_usage is not None:
            logger.info(
                f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
            )
            total_tokens = increase_token_usage(total_tokens, token_usage)
            logger.info(
                f"overall total tokens: {total_tokens['total_tokens']}, overall prompt tokens: {total_tokens['prompt_tokens']}, overall completion tokens: {total_tokens['completion_tokens']}"
            )
        if step_reasoning_process is not None:
            logger.info(f"\n\n{step_reasoning_process}\n\n")
        if step_output is not None:
            logger.info(step_output)

    return print_step

@pytest.fixture(scope="module")
def project_structure():
    return """
/home/ubuntu/projects/github/tabula-data/pk_prompt_list.md - f
/home/ubuntu/projects/github/tabula-data/README.md - f
/home/ubuntu/projects/github/tabula-data/app.py - f
/home/ubuntu/projects/github/tabula-data/pyproject.toml - f
/home/ubuntu/projects/github/tabula-data/models.log - f
/home/ubuntu/projects/github/tabula-data/poetry.lock - f
/home/ubuntu/projects/github/tabula-data/.env.swp - f
/home/ubuntu/projects/github/tabula-data/.gitignore - f
/home/ubuntu/projects/github/tabula-data/.dockerignore - f
/home/ubuntu/projects/github/tabula-data/app_script.py - f
/home/ubuntu/projects/github/tabula-data/docker-compose.yml - f
/home/ubuntu/projects/github/tabula-data/version.py - f
/home/ubuntu/projects/github/tabula-data/LICENSE - f
/home/ubuntu/projects/github/tabula-data/.bumpversion.cfg - f
/home/ubuntu/projects/github/tabula-data/Dockerfile - f
/home/ubuntu/projects/github/tabula-data/temp1.csv - f
/home/ubuntu/projects/github/tabula-data/.pre-commit-config.yaml - f
/home/ubuntu/projects/github/tabula-data/.env.template - f
/home/ubuntu/projects/github/tabula-data/scripts/extract_pk_summary_papers.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_header_categorize_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_split_by_col_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_drug_matching_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_llm.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_drug_info_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_llm_1.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_patient_info_refine_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_workflow.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/conftest.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_drug_match_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_TabFuncFlow.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_drug_info_agent.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_patient_matching_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_param_type_unit_extract_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_time_unit_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_row_cleanup_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_select_pk_tables.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_param_value_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_individual_data_del_step.py - f
/home/ubuntu/projects/github/tabula-data/system_tests/test_pk_sum_assembly_step.py - f
/home/ubuntu/projects/github/tabula-data/tests/17635501.html - f
/home/ubuntu/projects/github/tabula-data/tests/test_utils.py - f
/home/ubuntu/projects/github/tabula-data/tests/__init__.py - f
/home/ubuntu/projects/github/tabula-data/tests/test_get-html-content.py - f
/home/ubuntu/projects/github/tabula-data/tests/test_generated_table_processor.py - f
/home/ubuntu/projects/github/tabula-data/tests/conftest.py - f
/home/ubuntu/projects/github/tabula-data/tests/bak_test_full_text_vectordb.py - f
/home/ubuntu/projects/github/tabula-data/tests/test_concate_llm_contents.py - f
/home/ubuntu/projects/github/tabula-data/tests/test_prompts_utils.py - f
/home/ubuntu/projects/github/tabula-data/tests/test_stampers.py - f
/home/ubuntu/projects/github/tabula-data/tests/test_benchmark_evalulator.py - f
/home/ubuntu/projects/github/tabula-data/tests/30950674.html - f
/home/ubuntu/projects/github/tabula-data/tests/test_benchmark_pk_preprocess.py - f
/home/ubuntu/projects/github/tabula-data/tests/test_table-only-prompts.py - f
/home/ubuntu/projects/github/tabula-data/tests/test_table_extractor.py - f
/home/ubuntu/projects/github/tabula-data/tests/30950674_ds_result.json - f
/home/ubuntu/projects/github/tabula-data/tests/test_benchmark_utils.py - f
/home/ubuntu/projects/github/tabula-data/tests/35465728_ds_result.json - f
/home/ubuntu/projects/github/tabula-data/tests/data - d
"""