
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
from bioguider.agents.agent_utils import (
    convert_plan_to_string, 
    get_tool_names_and_descriptions,
    PlanAgentResultJsonSchema,
    PlanAgentResult,
)
from bioguider.agents.common_agent_2step import CommonAgentTwoChainSteps, CommonAgentTwoSteps
from bioguider.agents.peo_common_step import PEOCommonStep
from bioguider.agents.collection_task_utils import CollectionWorkflowState
from bioguider.agents.prompt_utils import COLLECTION_GOAL, COLLECTION_PROMPTS

COLLECTION_PLAN_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""### **Goal**  
You are an expert developer specializing in the biomedical domain. 
**{goal}**

{related_file_description}
---

### **Repository File Structure**  
Below is the 2-level file structure of the repository (`f` = file, `d` = directory, `l` - symlink, `u` - unknown):  
{repo_structure}

---

### **Function Tools**  
You have access to the following function tools:  
{tools}

---

### **Intermediate Steps**  
Here are the results from previous steps:  
{intermediate_steps}

---

### **Intermediate Thoughts**  
- **Analysis**: {intermediate_analysis}  
- **Thoughts**: {intermediate_thoughts}

---

### **Instructions**

1. We will iterate through multiple **Plan -> Execution -> Observation** loops as needed.  
   - All variables and tool outputs are **persisted across rounds**, so you can build on prior results.  
   - Develop your plan **incrementally**, and reflect on intermediate observations before proceeding.  
   - Limit each step to **one or two actions** — avoid trying to complete everything in a single step.

2. Your task is to collect all files that are relevant to the goal.  
   - Start by using the `summarize_file` tool to inspect file content quickly.  
   - If needed, follow up with the `read_file` tool for full content extraction.

3. You may use the `read_directory` tool to explore directory contents, but avoid using it in the first step unless necessary.

4. Your plan can only use the above tools, **do not** make up any tools not in the above tools list.

5. Your planned step input file or input directory must come from the above repository files structure, **do not** make up file name or directory name.

---

### **Important Instructions**
{important_instructions}

### **Output Format**  
Your plan **must exactly match** a sequence of steps in the following format, **do not** make up anything:

Step: <tool name>   # Tool name **must be one** of {tool_names}  
Step Input: <file or directory name>

Step: <tool name>  # Tool name **must be one** of {tool_names}  
Step Input: <file or directory name>
...
""")

class CollectionPlanStep(PEOCommonStep):
    """
    CollectionPlanStep is a step in the collection plan process.
    It is responsible for initializing the tools and compiling the step.
    """
    
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        repo_path: str,
        repo_structure: str,
        gitignore_path: str,
        custom_tools: list[BaseTool] | None = None,
    ):
        super().__init__(llm)
        self.step_name = "Collection Plan Step"
        self.repo_path = repo_path
        self.repo_structure = repo_structure
        self.gitignore_path = gitignore_path
        self.custom_tools = custom_tools if custom_tools is not None else []
    
        
    def _prepare_system_prompt(self, state: CollectionWorkflowState) -> str:
        collection_state = state
        goal_item = collection_state["goal_item"]
        collection_item = COLLECTION_PROMPTS[goal_item]
        intermediate_steps = self._build_intermediate_steps(state)
        step_analysis, step_thoughts = self._build_intermediate_analysis_and_thoughts(state)
        goal = ChatPromptTemplate.from_template(COLLECTION_GOAL).format(goal_item=collection_item["goal_item"])
        related_file_description = collection_item["related_file_description"]
        important_instructions="N/A" if "plan_important_instructions" not in collection_item or len(collection_item["plan_important_instructions"]) == 0 \
            else collection_item["plan_important_instructions"]
        tool_names, tools_desc = get_tool_names_and_descriptions(self.custom_tools)
        system_prompt = COLLECTION_PLAN_SYSTEM_PROMPT.format(
            goal=goal,
            related_file_description=related_file_description,
            repo_structure=self.repo_structure,
            tools=tools_desc,
            intermediate_steps=intermediate_steps,
            intermediate_analysis=step_analysis,
            intermediate_thoughts=step_thoughts,
            tool_names=tool_names,
            important_instructions=important_instructions,
        )
        self._print_step(
            state,
            step_output="**Intermediate Step Output**\n" + intermediate_steps
        )
        self._print_step(
            state,
            step_output="**Intermediate Step Analysis**\n{step_analysis}\n**Intermediate Step Thoughts**\n{step_thoughts}",
        )
        return system_prompt

    def _execute_directly(self, state: CollectionWorkflowState):
        system_prompt = self._prepare_system_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the collection plan step.",
            schema=PlanAgentResultJsonSchema,
        )
        PEOCommonStep._reset_step_state(state)
        res = PlanAgentResult(**res)
        self._print_step(state, step_output=f"**Reasoning Process**\n{reasoning_process}")
        self._print_step(state, step_output=f"**Plan**\n{str(res.actions)}")
        state["plan_actions"] = convert_plan_to_string(res)

        return state, token_usage
        
        
        





