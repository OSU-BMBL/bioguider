import os
from typing import Optional
# from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import pytest
from dotenv import load_dotenv
import logging

from bioguider.utils.constants import DEFAULT_TOKEN_USAGE
from bioguider.agents.agent_utils import increase_token_usage

load_dotenv()

def get_openai():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL_NAME"),
    )


def get_azure_openai():
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_MODEL", None),
        max_retries=5,
        # temperature=0.0,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
    )


def get_deepseek():
    """return ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        temperature=0.0,
        max_completion_tokens=10000,
        max_retries=3,
    )"""
    return None

@pytest.fixture(scope="module")
def llm():
    return get_azure_openai()

@pytest.fixture(scope="module")
def project_structure():
    return """
/home/ubuntu/projects/github/tabula-data/pk_prompt_list.md - f
/home/ubuntu/projects/github/tabula-data/README.md - f
/home/ubuntu/projects/github/tabula-data/app.py - f
/home/ubuntu/projects/github/tabula-data/pyproject.toml - f
/home/ubuntu/projects/github/tabula-data/models.log - f
/home/ubuntu/projects/github/tabula-data/poetry.lock - f
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
/home/ubuntu/projects/github/tabula-data/logs/test.log - f
/home/ubuntu/projects/github/tabula-data/logs/scripts.log - f
/home/ubuntu/projects/github/tabula-data/logs/test-20250329.log - f
/home/ubuntu/projects/github/tabula-data/logs/app.log - f
/home/ubuntu/projects/github/tabula-data/logs/test.log.bak - f
/home/ubuntu/projects/github/tabula-data/logs/app-20250329.log - f
/home/ubuntu/projects/github/tabula-data/logs/app-20250328.log - f
/home/ubuntu/projects/github/tabula-data/logs/app_scripts.log - f
/home/ubuntu/projects/github/tabula-data/logs/benchmark.log - f
/home/ubuntu/projects/github/tabula-data/logs/.gitkeep - f
/home/ubuntu/projects/github/tabula-data/extractor/make_request.py - f
/home/ubuntu/projects/github/tabula-data/extractor/table_utils.py - f
/home/ubuntu/projects/github/tabula-data/extractor/prompts_utils.py - f
/home/ubuntu/projects/github/tabula-data/extractor/request_openai.py - f
/home/ubuntu/projects/github/tabula-data/extractor/__init__.py - f
/home/ubuntu/projects/github/tabula-data/extractor/stampers.py - f
/home/ubuntu/projects/github/tabula-data/extractor/html_table_extractor.py - f
/home/ubuntu/projects/github/tabula-data/extractor/utils.py - f
/home/ubuntu/projects/github/tabula-data/extractor/article_retriever.py - f
/home/ubuntu/projects/github/tabula-data/extractor/request_geminiai.py - f
/home/ubuntu/projects/github/tabula-data/extractor/generated_table_processor.py - f
/home/ubuntu/projects/github/tabula-data/extractor/pk_sum_extractor.py - f
/home/ubuntu/projects/github/tabula-data/extractor/log_utils.py - f
/home/ubuntu/projects/github/tabula-data/extractor/request_deepseek.py - f
/home/ubuntu/projects/github/tabula-data/extractor/constants.py - f
/home/ubuntu/projects/github/tabula-data/extractor/agents - d
/home/ubuntu/projects/github/tabula-data/TabFuncFlow/TableFunctionFlowTutorial.ipynb - f
/home/ubuntu/projects/github/tabula-data/TabFuncFlow/steps_pk_summary - d
/home/ubuntu/projects/github/tabula-data/TabFuncFlow/operations - d
/home/ubuntu/projects/github/tabula-data/TabFuncFlow/utils - d
/home/ubuntu/projects/github/tabula-data/TabFuncFlow/steps_pk_individual - d
/home/ubuntu/projects/github/tabula-data/TabFuncFlow/pipelines - d
/home/ubuntu/projects/github/tabula-data/images/mprint-logo.png - f
/home/ubuntu/projects/github/tabula-data/images/favicon.png - f
/home/ubuntu/projects/github/tabula-data/images/copper-logo.png - f
/home/ubuntu/projects/github/tabula-data/prompts/pk_prompts_0107.json - f
/home/ubuntu/projects/github/tabula-data/prompts/pk_prompts.json - f
/home/ubuntu/projects/github/tabula-data/prompts/pe_prompts.json - f
/home/ubuntu/projects/github/tabula-data/prompts/cot_examples - d
/home/ubuntu/projects/github/tabula-data/prompts/chainprompts - d
/home/ubuntu/projects/github/tabula-data/benchmark/test_pk_summary_benchmark_with_semantic.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/test_pe_benchmark_with_llm.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/__init__.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/utils.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/test_pk_summary_benchmark_with_llm.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/pe_benchmark_with_semantic.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/comm_llm.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/pe_preprocess.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/comm_semantic.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/test_pk_summary_benchmark_with_semantic_specified_pmid.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/conftest.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/evaluate.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/pk_summary_benchmark_with_semantic.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/test_pe_benchmark_with_semantic.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/constant.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/pk_preprocess.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/common.py - f
/home/ubuntu/projects/github/tabula-data/benchmark/yichuan - d
/home/ubuntu/projects/github/tabula-data/benchmark/butils - d
/home/ubuntu/projects/github/tabula-data/benchmark/data - d
/home/ubuntu/projects/github/tabula-data/benchmark/logs - d
/home/ubuntu/projects/github/tabula-data/benchmark/result - d
/home/ubuntu/projects/github/tabula-data/components/__init__.py - f
/home/ubuntu/projects/github/tabula-data/components/main_tab.py - f
/home/ubuntu/projects/github/tabula-data/components/html_table_tab.py - f
"""


# ============================================================================================
# utils
@pytest.fixture(scope="session", autouse=True)
def prepare_logging():
    level = logging.INFO
    logging.basicConfig(level=level)
    file_handler = logging.FileHandler("./logs/test.log")
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

@pytest.fixture(scope="module")
def step_callback():
    total_tokens = {**DEFAULT_TOKEN_USAGE}

    def print_step(
        step_name: Optional[str] = None,
        step_output: Optional[str] = None,
        token_usage: Optional[dict] = None,
    ):
        nonlocal total_tokens
        logger = logging.getLogger(__name__)
        if step_name is not None:
            logger.info("=" * 64)
            logger.info(step_name)
        if token_usage is not None:
            logger.info(
                f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
            )
            total_tokens = increase_token_usage(total_tokens, token_usage)
            logger.info(
                f"overall total tokens: {total_tokens['total_tokens']}, overall prompt tokens: {total_tokens['prompt_tokens']}, overall completion tokens: {total_tokens['completion_tokens']}"
            )
        if step_output is not None:
            logger.info(step_output)

    return print_step


@pytest.fixture(scope="module")
def plan_actions():
    return 'Step: check_file_related_tool\nStep Input: pyproject.toml\nStep: check_file_related_tool\nStep Input: README.md\n'


