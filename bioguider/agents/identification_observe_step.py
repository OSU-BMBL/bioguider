
from langchain.prompts import ChatPromptTemplate

from bioguider.agents.agent_utils import ObservationResult
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps, CommonAgentTwoChainSteps
from bioguider.agents.identification_task_utils import IdentificationWorkflowState
from bioguider.agents.peo_common_step import PEOWorkflowState, PEOCommonStep
from bioguider.utils.constants import MAX_STEP_COUNT


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
Decide whether the goal can be achieved with the current information:
- If YES — fill `Analysis` and `FinalAnswer` (see Output Format below), leave `Thoughts` null.
- If NO — fill `Thoughts` explaining what is still missing, leave `Analysis` and
  `FinalAnswer` null. We will iterate in the next round.

Be precise and support your reasoning with evidence from the input.

### **Important Instructions**
{important_instructions}

### Notes
We are collecting information over multiple rounds; your thoughts and the output of this
step will be persisted, so **do not rush to provide a Final Answer**. If you find the
current information insufficient, share your reasoning under `Thoughts` and we will
continue with the next round accordingly.

### **Output Format (STRICT — read carefully)**
You MUST return a **single raw JSON object** — nothing else. No prose before or after,
no `<think>` blocks, no markdown, no ```json fences, no commentary.

The JSON object has exactly three optional fields:

  - `Analysis`   : string. Your reasoning when you ARE confident and ready to finalize.
  - `FinalAnswer`: string. A JSON-encoded string that matches the goal's expected shape.
                   For this goal, the shape is:
                   {final_answer_example}
                   Emit it as a **string** (use escaped quotes inside the JSON string).
                   Only fill this when you are confident the goal is fully achieved.
  - `Thoughts`   : string. Your reasoning when you are NOT yet ready — describe what is
                   missing and what you still need in the next round.

**Hard rules — any violation means your answer is wrong:**
  1. Return raw JSON only. No code fences, no leading/trailing text.
  2. Fill either (`Analysis` + `FinalAnswer`) OR (`Thoughts`). Not both.
  3. The value of `FinalAnswer` is a **string**, not a nested JSON object.
  4. Do NOT rush to finalize. If uncertain, use `Thoughts` and leave `FinalAnswer` null.

**Concrete example (not yet ready):**
{{"Analysis": null, "FinalAnswer": null, "Thoughts": "Still need to inspect pyproject.toml to confirm the project type before finalizing."}}
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
        step_count = state["step_count"]
        instruction = "Now, we have reached max recursion limit, please give me the **final answer** based on the current information" \
            if step_count == MAX_STEP_COUNT/3 - 2 else "Now, Let's begin."
        system_prompt = self._prepare_system_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=instruction,
            schema=ObservationResult,
        )
        state["final_answer"] = res.FinalAnswer
        analysis = res.Analysis
        thoughts = res.Thoughts
        state["step_analysis"] = analysis
        state["step_thoughts"] = thoughts
        state["step_count"] += 1
        self._print_step(
            state,
            step_output=f"**Observation Reasoning Process {state['step_count']}**\n{reasoning_process}"
        )
        self._print_step(
            state,
            step_output=f"Final Answer: {res.FinalAnswer if res.FinalAnswer else None}\nAnalysis: {analysis}\nThoughts: {thoughts}",
        )
        return state, token_usage