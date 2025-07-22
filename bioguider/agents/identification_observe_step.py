
from langchain.prompts import ChatPromptTemplate

from bioguider.agents.agent_utils import ObservationResult
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps, CommonAgentTwoChainSteps
from bioguider.agents.identification_task_utils import IdentificationWorkflowState
from bioguider.agents.peo_common_step import PEOWorkflowState, PEOCommonStep


## observation system prompt
IDENTIFICATION_OBSERVATION_SYSTEM_PROMPT = """Your goal is:
{goal}

### **Repository File Structure**
Here is the 2-level file structure of the repository (f - file, d - directory, l - symlink, u - unknown):
{repo_structure}

### **Intermediate Output**
{intermediate_output}

### **Instructions**
Carefully review the **Goal**, **Repository File Structure**, and **Intermediate Output**.
- If you believe the goal **can be achieved**, proceed as follows:  
  - Provide your reasoning under **Analysis**  
  - Then provide your result under **FinalAnswer**  
  ```
  **Analysis**: your analysis here  
  **FinalAnswer**: your final answer here, in **raw json format**, **including** the surrounding "{{}}" but **without** using code fence (```json ... ```), 
  For example, output exactly: {final_answer_example}
  ```
- If the information is **not sufficient** to achieve the goal, simply explain why under **Thoughts**:  
  ```
  **Thoughts**: your thoughts here
  ```
Be precise and support your reasoning with evidence from the input.

### **Important Instructions**
{important_instructions}

### Notes
We are collecting information over multiple rounds, your thoughts and the output of this step will be persisted, so please **do not rush to provide a Final Answer**.  
If you find the current information insufficient, share your reasoning or thoughts instead—we’ll continue with the next round accordingly.
"""


class IdentificationObserveStep(PEOCommonStep):
    def __init__(
        self,
        llm,
        repo_path: str,
        repo_structure: str,
        gitignore_path: str,
        custom_tools: list = None,
    ):
        super().__init__(llm)
        self.step_name = "Identification Observe Step"
        self.repo_path = repo_path
        self.repo_structure = repo_structure
        self.gitignore_path = gitignore_path
        self.custom_tools = custom_tools if custom_tools is not None else []

    def _prepare_system_prompt(self, state: IdentificationWorkflowState): 
        goal = state["goal"]
        important_instructions = "N/A" \
            if not "observe_instructions" in state else state["observe_instructions"]
        final_answer_example = state["final_answer_example"]
        intermediate_output = self._build_intermediate_steps(state)
        prompt = ChatPromptTemplate.from_template(IDENTIFICATION_OBSERVATION_SYSTEM_PROMPT)

        return prompt.format(
            goal=goal,
            repo_structure=self.repo_structure,
            intermediate_output=intermediate_output,
            final_answer_example=final_answer_example,
            important_instructions=important_instructions,
        )

    def _execute_directly(self, state: IdentificationWorkflowState):
        system_prompt = self._prepare_system_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin.",
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