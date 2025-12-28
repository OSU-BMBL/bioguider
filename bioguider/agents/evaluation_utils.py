import json
from pathlib import Path
from typing import Iterable, Tuple

from bioguider.agents.agent_utils import read_file
from bioguider.agents.consistency_evaluation_task import (
    ConsistencyEvaluationResult,
    ConsistencyEvaluationTask,
)
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE
from bioguider.utils.file_utils import detect_file_type
from bioguider.utils.notebook_utils import (
    extract_markdown_from_notebook,
    strip_notebook_to_code_and_markdown,
)
from bioguider.utils.pyphen_utils import PyphenReadability
from bioguider.utils.utils import convert_html_to_text
from .common_agent_2step import CommonAgentTwoChainSteps, CommonAgentTwoSteps


def _escape_template_braces(text: str) -> str:
    return text.replace("{", "<<").replace("}", ">>")


def sanitize_files(
    repo_path: str,
    files: Iterable[str],
    max_size_bytes: int,
    disallowed_exts: set[str] | None = None,
    check_ipynb_size: bool = False,
) -> list[str]:
    sanitized_files: list[str] = []
    for file in files:
        file_path = Path(repo_path, file)
        if not file_path.exists() or not file_path.is_file():
            continue
        if detect_file_type(file_path) == "binary":
            continue
        if disallowed_exts and file_path.suffix.lower() in disallowed_exts:
            continue
        if file_path.suffix.lower() != ".ipynb" or check_ipynb_size:
            if file_path.stat().st_size > max_size_bytes:
                continue
        sanitized_files.append(file)
    return sanitized_files


def normalize_evaluation_content(
    repo_path: str,
    file: str,
) -> Tuple[str | None, str | None]:
    file_path = Path(repo_path, file)
    content = read_file(file_path)
    if content is None:
        return None, None

    suffix = file_path.suffix.lower()
    if suffix == ".ipynb":
        readability_content = extract_markdown_from_notebook(file_path)
        content = json.dumps(strip_notebook_to_code_and_markdown(file_path))
        content = _escape_template_braces(content)
        return content, readability_content

    if suffix in {".html", ".htm"}:
        readability_content = convert_html_to_text(file_path)
        content = _escape_template_braces(readability_content)
        return content, readability_content

    return content, content


def compute_readability_metrics(
    content: str,
) -> Tuple[float, float, float, float]:
    readability = PyphenReadability()
    flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index, \
        _, _, _, _, _ = readability.readability_metrics(content)
    return flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index


def evaluate_consistency_on_content(
    llm,
    code_structure_db,
    step_callback,
    domain: str,
    content: str,
) -> Tuple[ConsistencyEvaluationResult | None, dict]:
    if code_structure_db is None:
        return None, {**DEFAULT_TOKEN_USAGE}
    consistency_evaluation_task = ConsistencyEvaluationTask(
        llm=llm,
        code_structure_db=code_structure_db,
        step_callback=step_callback,
    )
    return (
        consistency_evaluation_task.evaluate(
            domain=domain,
            documentation=content,
        ),
        {**DEFAULT_TOKEN_USAGE},
    )


def run_llm_evaluation(
    llm,
    system_prompt: str,
    instruction_prompt: str,
    schema,
    chain: bool = False,
) -> Tuple[object, dict, str | None]:
    agent_cls = CommonAgentTwoChainSteps if chain else CommonAgentTwoSteps
    agent = agent_cls(llm=llm)
    res, _processed, token_usage, reasoning_process = agent.go(
        system_prompt=system_prompt,
        instruction_prompt=instruction_prompt,
        schema=schema,
    )
    return res, token_usage, reasoning_process


def default_consistency_result(domain_label: str) -> ConsistencyEvaluationResult:
    return ConsistencyEvaluationResult(
        score=0,
        assessment=f"No sufficient information to evaluate the consistency of the {domain_label} documentation",
        development=[],
        strengths=[],
    )
