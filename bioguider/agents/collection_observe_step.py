
from typing import Callable
from langchain.tools import BaseTool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from bioguider.agents.agent_utils import ObservationResult
from bioguider.agents.collection_task_utils import CollectionWorkflowState
from bioguider.agents.common_agent_2step import CommonAgentTwoChainSteps, CommonAgentTwoSteps
from bioguider.agents.peo_common_step import PEOCommonStep
from bioguider.agents.prompt_utils import COLLECTION_GOAL, COLLECTION_PROMPTS


COLLECTION_OBSERVE_SYSTEM_PROMPT = """You are an expert software developer and technical documentation analyst.
{goal_item_desc}

{related_file_description}
---

### **Repository Structure**
Here is the 2-level file structure of the repository (`f` = file, `d` = directory, `l` - symlink, `u` - unknown):
{repo_structure}
---

### **Intermediate Output**
{intermediate_output}
---

### **Instructions**
1. Your goal is to identify files that are relevant to the **goal item**.
2. Carefully review the **Goal**, **Repository Structure**, and **Intermediate Output**.
3. If you believe **all relevant files** have been collected:

* Proceed with the following format:

  * Provide your reasoning under **Analysis**
  * Then list all relevant files and folders under **FinalAnswer**
  * **FinalAnswer** format must exactly match this format:
    **FinalAnswer**: {{"final_answer": [<file path>, <file path>, <file path>, ...]}}
  * Be sure to include the **full relative paths** with respect to the repository root.
  * Your answer **must exactly match the follwing format** (note: no JSON code block, no additional comments), **do not** make up anything:

  ```
  **Analysis**: your analysis here 
  **FinalAnswer**: {{"final_answer": ["path/to/file1", "path/to/file2", ...]}}
  ```
4. If you believe **more files still need to be collected**:
* Provide your reasoning under **Thoughts**:

  ```
  **Thoughts**: your explanation here
  ```
  
5. Important instructions:
  {important_instructions}
Be precise and support your reasoning with evidence from the input.
---

### Notes
- We are collecting information over multiple rounds, your thoughts and the output of this step will be persisted, so please **do not rush to provide a Final Answer**.  
  If you find the current information insufficient, share your thoughts instead—we’ll continue with the next round accordingly.
"""

class CollectionObserveStep(PEOCommonStep):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        repo_path: str,
        repo_structure: str,
        gitignore_path: str,
    ):
        super().__init__(llm=llm)
        self.repo_path = repo_path
        self.repo_structure = repo_structure
        self.gitignore_path = gitignore_path
        self.step_name = "Collection Observation Step"

    def _build_prompt(self, state):
        str_goal_item = state["goal_item"]
        collection_item = COLLECTION_PROMPTS[str_goal_item]
        goal_item_desc = \
            ChatPromptTemplate.from_template(COLLECTION_GOAL).format(goal_item=collection_item["goal_item"])
        repo_structure = self.repo_structure
        intermediate_steps = self._build_intermediate_steps(state)
        prompt = ChatPromptTemplate.from_template(COLLECTION_OBSERVE_SYSTEM_PROMPT)
        important_instructions = "N/A" if "observe_important_instructions" not in collection_item or len(collection_item["observe_important_instructions"]) == 0 \
            else collection_item["observe_important_instructions"]
        return prompt.format(
            goal_item_desc=goal_item_desc,
            related_file_description=collection_item["related_file_description"],
            repo_structure=repo_structure,
            intermediate_output=intermediate_steps,
            important_instructions=important_instructions,
        )
    def _execute_directly(self, state: CollectionWorkflowState):
        system_prompt = self._build_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Let's begin thinking.",
            schema=ObservationResult,
        )
        state["final_answer"] = res.FinalAnswer
        analysis = res.Analysis
        thoughts = res.Thoughts
        state["step_analysis"] = analysis
        state["step_thoughts"] = thoughts
        self._print_step(
            state,
            step_output=f"**Observation Reasoning Process**\n{reasoning_process}"
        )
        self._print_step(
            state,
            step_output=f"Final Answer: {res.FinalAnswer if res.FinalAnswer else None}\nAnalysis: {analysis}\nThoughts: {thoughts}",
        )
        return state, token_usage
