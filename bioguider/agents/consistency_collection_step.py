



from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState
from bioguider.agents.peo_common_step import PEOCommonStep


CONSISTANCY_COLLECTION_SYSTEM_PROMPT = """
### **Goal**
You are an expert developer specializing in the biomedical domain.
You will be given a {domain} documentation. Your task has **two parts**:

1. Collect all the functions, classes, and methods that the {domain} documentation mentions.
2. Collect all the **command-line invocations** that the {domain} documentation shows
   (for example `python scripts/run.py -f aaa.dat -b -o out/` or
   `Rscript bin/analyze.R --input data.csv --threshold 0.05`).
   Look inside shell / console / bash fenced code blocks and at any inline commands.
   Copy the program name and every flag **verbatim** from the documentation.
   **Do not** make up anything; when you are not sure about a field, put `"N/A"`.

---

### **Input {domain} Documentation**
{documentation}

### **Output Format**

`functions_and_classes` — each item:
```
name: <function/class/method name>
file_path: <file path, if not sure, just put "N/A">
parameters: <parameters, if not sure, just put "N/A">
parent: <parent name, if it is a class method, put the class name as the parent name, if not sure, just put "N/A">
```

`cli_invocations` — each item:
```
program: <the executable or script as written, e.g. "run.py", "analyze.R", "mytool", "python -m pkg.cli"; if not sure, "N/A">
language: <"python" if it is run via python / a .py script, "r" if run via Rscript / R CMD / a .R script, "shell" for a plain shell command, otherwise "unknown">
subcommand: <sub-command name if the tool uses one, e.g. "train"; otherwise "N/A">
options: <a list of the options/flags/positional arguments shown; each entry has:
            name  - the flag exactly as written ("-f", "--file"), or "<positional>" for a positional argument
            value - the value shown for that option, or "N/A" if none is shown
            kind  - "option" (takes a value), "flag" (boolean switch, no value), or "positional">
source: <the exact command line as written in the documentation>
```

The collected items **must exactly match** the formats above — **do not** make anything up.

---

### **Output Example**
`functions_and_classes`:
```
name: __init__
file_path: src/agents/common_agent.py
parameters: llm, step_output_callback, summarized_files_db
parent: CommonAgent

name: _invoke_agent
file_path: src/agents/common_agent.py
parameters: system_prompt, instruction_prompt, schema, post_process
parent: CommonAgent

...
```

`cli_invocations`:
```
program: scripts/run.py
language: python
subcommand: N/A
options:
  - name: -f
    value: aaa.dat
    kind: option
  - name: -b
    value: N/A
    kind: flag
  - name: -o
    value: out/
    kind: option
source: python scripts/run.py -f aaa.dat -b -o out/

program: bin/analyze.R
language: r
subcommand: N/A
options:
  - name: --input
    value: data.csv
    kind: option
  - name: --threshold
    value: 0.05
    kind: option
source: Rscript bin/analyze.R --input data.csv --threshold 0.05

...
```

If the documentation contains no command-line invocations, return an empty list for `cli_invocations`.

"""

class ConsistencyCollectionResult(BaseModel):
    functions_and_classes: list[dict] = Field(
        default_factory=list,
        description="A list of functions and classes that the documentation mentions",
    )
    cli_invocations: list[dict] = Field(
        default_factory=list,
        description="A list of command-line invocations (program + options) that the documentation shows",
    )

ConsistencyCollectionResultJsonSchema = {
  "title": "ConsistencyCollectionResult",
  "description": "Collection of functions/classes and command-line invocations mentioned in the documentation",
  "properties": {
    "functions_and_classes": {
      "description": "A list of functions and classes that the documentation mentions",
      "items": {
        "type": "object"
      },
      "title": "Functions And Classes",
      "type": "array"
    },
    "cli_invocations": {
      "description": "A list of command-line invocations (program + options) that the documentation shows",
      "items": {
        "type": "object"
      },
      "title": "Cli Invocations",
      "type": "array"
    }
  },
  "required": [
    "functions_and_classes",
    "cli_invocations"
  ],
  "type": "object"
}

class ConsistencyCollectionStep(PEOCommonStep):
    def __init__(self, llm: BaseChatOpenAI):
        super().__init__(llm)
        self.step_name = "Consistency Collection Step"

    def _prepare_system_prompt(self, state: ConsistencyEvaluationState) -> str:
        documentation = state["documentation"]
        domain = state["domain"]
        return ChatPromptTemplate.from_template(CONSISTANCY_COLLECTION_SYSTEM_PROMPT).format(
            domain=domain,
            documentation=documentation,
        )

    def _execute_directly(self, state: ConsistencyEvaluationState) -> tuple[dict, dict[str, int]]:
        system_prompt = self._prepare_system_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the consistency collection step.",
            schema=ConsistencyCollectionResultJsonSchema,
        )
        res: ConsistencyCollectionResult = ConsistencyCollectionResult.model_validate(res)
        state["functions_and_classes"] = res.functions_and_classes
        state["cli_invocations"] = res.cli_invocations
        self._print_step(state, step_output=f"Consistency Collection Result (functions/classes): {res.functions_and_classes}")
        self._print_step(state, step_output=f"Consistency Collection Result (cli invocations): {res.cli_invocations}")
        self._print_step(state, step_output=f"Consistency Collection Reasoning Process: {reasoning_process}")

        return state, token_usage
