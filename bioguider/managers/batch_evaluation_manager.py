import logging
from typing import Callable, Optional

from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.utils.constants import (
    DEFAULT_TOKEN_USAGE,
    BatchRepoEvaluationResult,
    EvaluationStepEnum,
    EvaluationStepResult,
    StepStatus,
)
from bioguider.utils.utils import convert_to_serializable, increase_token_usage
from bioguider.managers.evaluation_manager import EvaluationManager

logger = logging.getLogger(__name__)

ALL_STEPS = [
    EvaluationStepEnum.identify,
    EvaluationStepEnum.readme,
    EvaluationStepEnum.installation,
    EvaluationStepEnum.userguide,
    EvaluationStepEnum.tutorial,
    # EvaluationStepEnum.submission_requirements,
]


def evaluate_repository(
    llm: BaseChatOpenAI,
    repo_url: str,
    step_callback: Optional[Callable] = None,
    steps: Optional[list[EvaluationStepEnum]] = None,
) -> BatchRepoEvaluationResult:
    """
    Run a full evaluation pipeline on a single repository.

    This is the primary API for batch evaluation. The caller (e.g. FastAPI server)
    is responsible for concurrency — this function is synchronous and blocking.

    Each evaluation step is isolated: if one step fails, the others still run.
    The submission_requirements step depends on readme and installation results,
    so it is skipped if either of those failed.

    Args:
        llm: LangChain-compatible chat model.
        repo_url: GitHub URL of the repository to evaluate.
        step_callback: Optional callback(step_name=, step_output=, token_usage=)
            for progress reporting.
        steps: Which evaluation steps to run. Defaults to all steps.

    Returns:
        BatchRepoEvaluationResult with per-step results, token usage, and errors.
    """
    if steps is None:
        steps = list(ALL_STEPS)

    result = BatchRepoEvaluationResult(
        repo_url=repo_url,
        status=StepStatus.running.value,
        steps={
            s.value: EvaluationStepResult(step=s.value)
            for s in steps
        },
    )

    mgr = EvaluationManager(llm, step_callback)

    # --- prepare repo (required for all steps) ---
    try:
        _report_step(step_callback, "prepare_repo")
        mgr.prepare_repo(repo_url)
    except Exception as e:
        logger.exception(f"Failed to prepare repo {repo_url}")
        result.status = StepStatus.failed.value
        result.error = f"prepare_repo failed: {e}"
        return result

    # --- identify project ---
    metadata = None
    if EvaluationStepEnum.identify in steps:
        metadata = _run_step(
            result,
            EvaluationStepEnum.identify,
            lambda: _do_identify(mgr),
            step_callback,
        )

    # --- readme ---
    readme_evaluation = None
    readme_files = None
    if EvaluationStepEnum.readme in steps:
        readme_result = _run_step(
            result,
            EvaluationStepEnum.readme,
            lambda: _do_readme(mgr),
            step_callback,
        )
        if readme_result is not None:
            readme_evaluation, readme_files = readme_result

    # --- installation ---
    installation_evaluation = None
    installation_files = None
    if EvaluationStepEnum.installation in steps:
        install_result = _run_step(
            result,
            EvaluationStepEnum.installation,
            lambda: _do_installation(mgr),
            step_callback,
        )
        if install_result is not None:
            installation_evaluation, installation_files = install_result

    # --- userguide ---
    if EvaluationStepEnum.userguide in steps:
        _run_step(
            result,
            EvaluationStepEnum.userguide,
            lambda: _do_userguide(mgr),
            step_callback,
        )

    # --- tutorial ---
    if EvaluationStepEnum.tutorial in steps:
        _run_step(
            result,
            EvaluationStepEnum.tutorial,
            lambda: _do_tutorial(mgr),
            step_callback,
        )

    # --- submission requirements (depends on readme + installation) ---
    if EvaluationStepEnum.submission_requirements in steps:
        if readme_evaluation is not None and installation_evaluation is not None:
            _run_step(
                result,
                EvaluationStepEnum.submission_requirements,
                lambda: _do_submission_requirements(
                    mgr, readme_evaluation, installation_files, installation_evaluation
                ),
                step_callback,
            )
        else:
            step_result = result.steps[EvaluationStepEnum.submission_requirements.value]
            step_result.status = StepStatus.skipped.value
            step_result.error = "Skipped: requires both readme and installation results"

    # --- finalize ---
    has_failure = any(
        s.status == StepStatus.failed.value
        for s in result.steps.values()
    )
    result.status = "completed_with_errors" if has_failure else StepStatus.completed.value

    total = {**DEFAULT_TOKEN_USAGE}
    for s in result.steps.values():
        total = increase_token_usage(total, s.token_usage)
    result.total_token_usage = total

    return result


# --- step runners ---

def _run_step(
    result: BatchRepoEvaluationResult,
    step_enum: EvaluationStepEnum,
    fn: Callable,
    step_callback: Optional[Callable],
):
    step_key = step_enum.value
    step_result = result.steps[step_key]
    step_result.status = StepStatus.running.value
    _report_step(step_callback, step_key)

    try:
        return_value = fn()
        step_result.status = StepStatus.completed.value

        if step_enum == EvaluationStepEnum.identify:
            result.project_metadata = convert_to_serializable(return_value)
            step_result.evaluation = result.project_metadata
        elif isinstance(return_value, tuple) and len(return_value) == 2:
            evaluation, files = return_value
            step_result.evaluation = convert_to_serializable(evaluation)
            step_result.files = files
        else:
            step_result.evaluation = convert_to_serializable(return_value)

        return return_value

    except Exception as e:
        logger.exception(f"Step {step_key} failed for {result.repo_url}")
        step_result.status = StepStatus.failed.value
        step_result.error = str(e)
        return None


def _report_step(step_callback: Optional[Callable], step_name: str):
    if step_callback is not None:
        step_callback(step_name=step_name, step_output=None, token_usage=None)


# --- individual step implementations ---

def _do_identify(mgr: EvaluationManager):
    return mgr.identify_project()


def _do_readme(mgr: EvaluationManager):
    return mgr.evaluate_readme()


def _do_installation(mgr: EvaluationManager):
    return mgr.evaluate_installation()


def _do_userguide(mgr: EvaluationManager):
    return mgr.evaluate_userguide()


def _do_tutorial(mgr: EvaluationManager):
    return mgr.evaluate_tutorial()


def _do_submission_requirements(
    mgr: EvaluationManager,
    readme_evaluation,
    installation_files,
    installation_evaluation,
):
    return mgr.evaluate_submission_requirements(
        readme_files_evaluation=readme_evaluation,
        installation_files=installation_files,
        installation_evaluation=installation_evaluation,
    )
