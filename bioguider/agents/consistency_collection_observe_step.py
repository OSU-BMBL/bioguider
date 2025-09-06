from typing import Callable

from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from bioguider.agents.agent_utils import ObservationResult
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.consistency_collection_task_utils import ConsistencyCollectionWorkflowState
from bioguider.agents.peo_common_step import PEOCommonStep

CONSISTENCY_EVAL_OBSERVE_SYSTEM_PROMPT = """You are an expert developer specializing in the biomedical domain.

### **Goal**
Your task is to collect the function, class, and method definitions and docstrings for a given user guide/API documentation.

---

### **Intermediate Steps**  
Here are the results from previous steps:  
{intermediate_steps}

---

### **Instructions**
1. Your goal is if you have enough information to evaluate the consistency of the user guide/API documentation.
2. Carefully review the **Goal**, **User Guide/API Documentation**, and **Intermediate Output**.
3. If you believe you have enough information to evaluate the consistency of the user guide/API documentation:

* Proceed with the following format:

  * Provide your reasoning under **Analysis**
  * Then give your final answer under **FinalAnswer**
  * **FinalAnswer** format must exactly match this format:
    **FinalAnswer**: {{"final_answer": "yes" or "no"}}
  * Your answer **must exactly match the follwing format** (note: no JSON code block, no additional comments), **do not** make up anything:

  ```
  **Analysis**: your analysis here 
  **FinalAnswer**: {{"final_answer": "yes" or "no"}}
  ```
4. If you believe you do not have enough information to evaluate the consistency of the user guide/API documentation:

* Provide your reasoning under **Thoughts**:

  ```
  **Thoughts**: your thoughts here
  ```

Be precise and support your reasoning with evidence from the input.

---

### **Notes**
We are collecting information over multiple rounds, your thoughts and the output of this step will be persisted, so please **do not rush to provide a Final Answer**.  
If you find the current information insufficient, share your reasoning or thoughts instead—we’ll continue with the next round accordingly.

---

### **Input User Guide/API Documentation**
{user_guide_api_documentation}

---

"""


class ConsistencyCollectionObserveStep(PEOCommonStep):
    def __init__(
        self,
        llm: BaseChatOpenAI,
    ):
        super().__init__(llm=llm)
        self.step_name = "Consistency Collection Observe Step"

    def _build_prompt(self, state):
        user_guide_api_documentation = state["user_guide_api_documentation"]
        intermediate_steps = self._build_intermediate_steps(state)
        prompt = ChatPromptTemplate.from_template(CONSISTENCY_EVAL_OBSERVE_SYSTEM_PROMPT)
        return prompt.format(
            user_guide_api_documentation=user_guide_api_documentation,
            intermediate_steps=intermediate_steps,
        )

    def _collect_final_answer(self, state: ConsistencyCollectionWorkflowState):
        if not ("final_answer" in state and state["final_answer"] is not None and
            state["final_answer"].strip().lower() == "yes"):
            return None
        
        final_result = ""
        if "intermediate_steps" in state and state["intermediate_steps"] is not None:
            for i in range(len(state["intermediate_steps"])):
                final_result += state["intermediate_steps"][i]
                final_result += "\n\n"
        if "step_output" in state and state["step_output"] is not None:
            final_result += state["step_output"]
            final_result += "\n\n"
        
        return final_result

            
    def _execute_directly(self, state: ConsistencyCollectionWorkflowState):
        system_prompt = self._build_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the consistency collection observe step.",
            schema=ObservationResult,
        )
        state["final_answer"] = res.FinalAnswer
        analysis = res.Analysis
        thoughts = res.Thoughts
        state["step_analysis"] = analysis
        state["step_thoughts"] = thoughts
        state["step_count"] += 1
        state["final_assembly_result"] = self._collect_final_answer(state)
        self._print_step(
            state,
            step_output=f"**Observation Reasoning Process {state['step_count']}**\n{reasoning_process}"
        )
        self._print_step(
            state,
            step_output=f"Final Answer: {res.FinalAnswer if res.FinalAnswer else None}\nAnalysis: {analysis}\nThoughts: {thoughts}",
        )
        if state["final_assembly_result"] is not None:
            self._print_step(
                state,
                step_output=f"Final Assembly Result: {state['final_assembly_result']}",
            )
        return state, token_usage
