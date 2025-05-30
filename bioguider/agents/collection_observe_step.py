
from typing import Callable
from langchain.tools import BaseTool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from bioguider.agents.agent_utils import ObservationResult
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
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
1. Your goal is to identify files that are relevant to the goal item.
2. Carefully review the **Goal**, **Repository Structure**, and **Intermediate Output**.
- If you believe all files relevant to the goal item have been collected, proceed as follows:
  - Provide your reasoning under **Analysis**
  - Then provide all relevant files under **FinalAnswer**
  - Be sure to include the **full relative paths of the files** with respect to the repository root.
  ```
  **Analysis**: your analysis here
  **FinalAnswer**: your final answer here
  ```
- If you believe there are still files that need to be collected, simply explain why under **Thoughts**:
  ```
  **Thoughts**: your thoughts here
  ```
3. Important instructions:
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
        important_instructions = "N/A" if "important_instructions" not in collection_item or len(collection_item["important_instructions"]) == 0 \
            else collection_item["important_instructions"]
        return prompt.format(
            goal_item_desc=goal_item_desc,
            related_file_description=collection_item["related_file_description"],
            repo_structure=repo_structure,
            intermediate_output=intermediate_steps,
            important_instructions=important_instructions,
        )
    def _execute_directly(self, state):
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
