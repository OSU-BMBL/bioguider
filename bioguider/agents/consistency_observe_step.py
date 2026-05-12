

from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState
from bioguider.agents.peo_common_step import PEOCommonStep

CONSISTENCY_OBSERVE_SYSTEM_PROMPT = """
You are an expert developer specializing in the biomedical domain.
Your task is to analyze:
1. the {domain} documentation,
2. the code definitions (functions / classes / methods) related to the {domain} documentation,
3. the command-line-interface findings — i.e. how command-line invocations shown in the
   {domain} documentation compare with the command-line options actually defined in the code,
and generate a structured consistency assessment based on the following criteria.

---

### **Evaluation Criteria**

**Consistency**:
  * **Score**: [a number between 0 and 100 representing the consistency quality rating.]
  * **Assessment**: [Your evaluation of whether the {domain} documentation is consistent with the code definitions and the command-line interface.]
  * **Development**: [A list of inconsistencies — inconsistent function/class/method names, inconsistent docstrings, AND inconsistent command-line usage (undefined options, undefined sub-commands, command-line programs that don't exist in the codebase). Describe each one as specifically as possible.]
  * **Strengths**: [A list of strengths of the {domain} documentation on consistency.]

When using the command-line-interface findings:
  * Treat an option, sub-command, or program flagged as **not defined / not found** as a real
    consistency defect — the same severity as a wrong function name.
  * Do **not** penalize an invocation whose status is `not checked` (it is an R or other
    non-Python entry point whose options are not indexed yet); you may mention it neutrally.
  * If there are no command-line-interface findings, simply do not comment on the command line.

---

### **Output Format**
Your output **must exactly match** the following format:
```
**Consistency**:
  * **Score**: [a number between 0 and 100 representing the consistency quality rating.]
  * **Assessment**: [Your evaluation of whether the {domain} documentation is consistent with the code definitions and the command-line interface.]
  * **Development**: [A list of inconsistencies, please be as specific as possible]
  * **Strengths**: [A list of strengths of the {domain} documentation on consistency]
```

### **Output Example**

```
**Consistency**:
  * **Score**: [a number between 0 and 100 representing the consistency quality rating.]
  * **Assessment**: [Your evaluation ...]
  * **Development**:
    - Inconsistent function/class/method name 1
    - Inconsistent docstring 1
    - Documentation runs `python scripts/train.py --workers 4`, but the parser defines no `--workers` option
    - ...
  * **Strengths**:
    - Strengths 1
    - Strengths 2
    - ...
```

---

### **Input {domain} Documentation**
{documentation}

### **Code Definitions**
{code_definitions}

### **Command-Line Interface Findings**
{cli_findings}


"""

class ConsistencyEvaluationObserveResult(BaseModel):
    consistency_score: int=Field(description="A number between 0 and 100 representing the consistency quality rating.")
    consistency_assessment: str=Field(description="Your evaluation of whether the documentation is consistent with the code definitions")
    consistency_development: list[str]=Field(description="A list of inconsistent function/class/method names, docstrings, and command-line usages")
    consistency_strengths: list[str]=Field(description="A list of strengths of the documentation on consistency")


_CLI_STATUS_LABELS = {
    "ok": "consistent with the code",
    "issues": "INCONSISTENT with the code (see issues below)",
    "program_not_found": "the program is NOT FOUND in the codebase",
    "language_not_indexed": "not checked (non-Python entry point; options not indexed yet)",
}


def _format_cli_findings(cli_query_rows: list[dict[str, Any]] | None) -> str:
    """Render the query step's CLI findings into a prompt-friendly text block."""
    if not cli_query_rows:
        return "No command-line invocations were found in the documentation."
    blocks: list[str] = []
    for f in cli_query_rows:
        status = str(f.get("status", "unknown"))
        lines = [
            f"program (as written in the documentation): {f.get('program', 'N/A')}",
            f"command line shown: {f.get('source', 'N/A')}",
            f"status: {status} — {_CLI_STATUS_LABELS.get(status, status)}",
        ]
        if f.get("subcommand"):
            lines.append(f"sub-command used: {f['subcommand']}")
        if f.get("resolved_paths"):
            lines.append(f"matched code file(s): {', '.join(f['resolved_paths'])}")
        if f.get("defined_options"):
            lines.append(f"options the parser defines: {', '.join(f['defined_options'])}")
        if f.get("matched_options"):
            lines.append(f"documented options that match the parser: {', '.join(f['matched_options'])}")
        if f.get("unknown_options"):
            lines.append(f"documented options the parser does NOT define: {', '.join(f['unknown_options'])}")
        issues = f.get("issues") or []
        if issues:
            lines.append("issues:")
            lines.extend(f"  - {msg}" for msg in issues)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class ConsistencyObserveStep(PEOCommonStep):
    def __init__(self, llm: BaseChatOpenAI):
        super().__init__(llm)
        self.step_name = "Consistency Observe Step"

    def _prepare_system_prompt(self, state: ConsistencyEvaluationState):
        all_query_rows = state["all_query_rows"]
        documentation = state["documentation"]
        domain = state["domain"]
        code_definition = ""
        for row in all_query_rows:
            mismatch_note = ""
            if row.get("possible_name_mismatch"):
                doc_ref = row.get("doc_referenced_as", "unknown")
                match_type = row.get("match_type", "unknown")
                mismatch_note = (
                    f"\n[NOTE: The documentation referenced '{doc_ref}' but no exact match "
                    f"was found. This entry is a fuzzy match ({match_type}) for the actual "
                    f"name '{row['name']}'. Possible name error in the documentation.]"
                )
            content = (
                f"name: {row['name']}\nfile_path: {row['path']}\nparent: {row['parent']}\n"
                f"parameters: {row['params']}\ndoc_string: {row['doc_string']}{mismatch_note}"
            )
            code_definition += content
            code_definition += "\n\n\n"
        cli_findings = _format_cli_findings(state.get("cli_query_rows"))
        return ChatPromptTemplate.from_template(CONSISTENCY_OBSERVE_SYSTEM_PROMPT).format(
            code_definitions=code_definition,
            cli_findings=cli_findings,
            documentation=documentation,
            domain=domain,
        )

    def _execute_directly(self, state: ConsistencyEvaluationState):
        system_prompt = self._prepare_system_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the consistency evaluation step.",
            schema=ConsistencyEvaluationObserveResult,
        )
        res: ConsistencyEvaluationObserveResult = res
        state["consistency_score"] = res.consistency_score
        state["consistency_assessment"] = res.consistency_assessment
        state["consistency_development"] = res.consistency_development
        state["consistency_strengths"] = res.consistency_strengths
        return state, token_usage
