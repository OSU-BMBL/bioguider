

from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState
from bioguider.agents.peo_common_step import PEOCommonStep

CONSISTENCY_OBSERVE_SYSTEM_PROMPT = """
You are an expert developer specializing in the biomedical domain.
Your task is to analyze both:
1. the provided file related to {domain} documentation,
2. the code definitions related to the {domain} documentation
and generate a structured consistency assessment based on the following criteria.

---

### **Evaluation Criteria**

**Consistency**:
  * **Score**: [a number between 0 and 100 representing the consistency quality rating.]
  * **Assessment**: [Your evaluation of whether the {domain} documentation is consistent with the code definitions]
  * **Development**: [A list of inconsistent function/class/method name and inconsistent docstring, and describe how they are inconsistent, please be as specific as possible]
  * **Strengths**: [A list of strengths of the {domain} documentation on consistency]

---

### **Output Format**
Your output **must exactly match** the following format:
```
**Consistency**:
  * **Score**: [a number between 0 and 100 representing the consistency quality rating.]
  * **Assessment**: [Your evaluation of whether the {domain} documentation is consistent with the code definitions]
  * **Development**: [A list of inconsistent function/class/method name and inconsistent docstring, and describe how they are inconsistent, please be as specific as possible]
  * **Strengths**: [A list of strengths of the {domain} documentation on consistency]
```

### **Output Example**

```
**Consistency**:
  * **Score**: [a number between 0 and 100 representing the consistency quality rating.]
  * **Assessment**: [Your evaluation of whether the {domain} documentation is consistent with the code definitions]
  * **Development**:
    - Inconsistent function/class/method name 1
    - Inconsistent docstring 1
    - Inconsistent function/class/method name 2
    - Inconsistent docstring 2
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


"""

class ConsistencyEvaluationObserveResult(BaseModel):
    consistency_score: int=Field(description="A number between 0 and 100 representing the consistency quality rating.")
    consistency_assessment: str=Field(description="Your evaluation of whether the documentation is consistent with the code definitions")
    consistency_development: list[str]=Field(description="A list of inconsistent function/class/method name and inconsistent docstring")
    consistency_strengths: list[str]=Field(description="A list of strengths of the documentation on consistency")


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
            content = f"name: {row['name']}\nfile_path: {row['path']}\nparent: {row['parent']}\nparameters: {row['params']}\ndoc_string: {row['doc_string']}"
            code_definition += content
            code_definition += "\n\n\n"
        return ChatPromptTemplate.from_template(CONSISTENCY_OBSERVE_SYSTEM_PROMPT).format(
            code_definitions=code_definition,
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


