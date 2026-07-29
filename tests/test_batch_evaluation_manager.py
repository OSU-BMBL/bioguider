import json
import pytest

from bioguider.utils.constants import (
    DEFAULT_TOKEN_USAGE,
    BatchRepoEvaluationResult,
    EvaluationStepEnum,
    EvaluationStepResult,
    StepStatus,
)
from bioguider.managers.batch_evaluation_manager import ALL_STEPS


def test_batch_repo_evaluation_result_serializable():
    result = BatchRepoEvaluationResult(
        repo_url="https://github.com/example/repo",
        status=StepStatus.completed.value,
        project_metadata={"repo_name": "repo", "owner": "example"},
        steps={
            "readme": EvaluationStepResult(
                step="readme",
                status=StepStatus.completed.value,
                evaluation={"overall_score": 75},
                files=["README.md"],
                token_usage={"total_tokens": 100, "completion_tokens": 50, "prompt_tokens": 50},
            ),
            "installation": EvaluationStepResult(
                step="installation",
                status=StepStatus.failed.value,
                error="Some LLM error",
            ),
        },
    )
    json_str = result.model_dump_json(indent=2)
    parsed = json.loads(json_str)
    assert parsed["repo_url"] == "https://github.com/example/repo"
    assert parsed["status"] == "completed"
    assert parsed["steps"]["readme"]["status"] == "completed"
    assert parsed["steps"]["readme"]["evaluation"]["overall_score"] == 75
    assert parsed["steps"]["installation"]["status"] == "failed"
    assert parsed["steps"]["installation"]["error"] == "Some LLM error"


def test_evaluation_step_result_defaults():
    step = EvaluationStepResult(step="readme")
    assert step.status == "pending"
    assert step.evaluation is None
    assert step.files is None
    assert step.error is None
    assert step.token_usage == {**DEFAULT_TOKEN_USAGE}


def test_all_steps_is_subset_of_enums():
    all_values = {s.value for s in EvaluationStepEnum}
    step_values = {s.value for s in ALL_STEPS}
    assert step_values.issubset(all_values)
    assert len(ALL_STEPS) > 0


def test_step_status_values():
    assert StepStatus.pending.value == "pending"
    assert StepStatus.running.value == "running"
    assert StepStatus.completed.value == "completed"
    assert StepStatus.failed.value == "failed"
    assert StepStatus.skipped.value == "skipped"
