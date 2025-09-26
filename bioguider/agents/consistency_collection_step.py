


from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState
from bioguider.agents.peo_common_step import PEOCommonStep


CONSISTANCY_COLLECTION_SYSTEM_PROMPT = """
### **Goal**  
You are an expert developer specializing in the biomedical domain.
You will be given a {domain} documentation. Your task is to collect all the functions, classes, and methods that the {domain} documentation mentions.

---

### **Input {domain} Documentation**
{documentation}

### **Output Format**
The collected functions, classes, and methods **must exactly match** the following format, **do not** make up anything:

```
name: <function/class/method name>
file_path: <file path, if not sure, just put "N/A">
parameters: <parameters, if not sure, just put "N/A">
parent: <parent name, if it is a class method, put the class name as the parent name, if not sure, just put "N/A">

...

```

---

### **Output Example**
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

"""

class ConsistencyCollectionResult(BaseModel):
    functions_and_classes: list[dict] = Field(description="A list of functions and classes that the documentation mentions")

ConsistencyCollectionResultJsonSchema = {
  "properties": {
    "functions_and_classes": {
      "description": "A list of functions and classes that the documentation mentions",
      "items": {
        "type": "object"
      },
      "title": "Functions And Classes",
      "type": "array"
    }
  },
  "required": [
    "functions_and_classes"
  ],
  "title": "ConsistencyCollectionResult",
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
        self._print_step(state, step_output=f"Consistency Collection Result: {res.functions_and_classes}")
        self._print_step(state, step_output=f"Consistency Collection Reasoning Process: {reasoning_process}")
        
        return state, token_usage

