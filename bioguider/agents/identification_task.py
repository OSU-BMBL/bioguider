
import os
import re
import logging
from enum import Enum
from typing import Callable, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import tool, Tool
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from bioguider.agents.agent_tools import read_directory_tool, read_file_tool, summarize_file_tool
from bioguider.agents.agent_utils import (
    CustomOutputParser, 
    CustomPromptTemplate,
    ObservationResult, 
    read_directory,
    get_tool_names_and_descriptions,
)
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.prompt_utils import (
    IDENTIFICATION_EXECUTION_SYSTEM_PROMPT, 
    IDENTIFICATION_GOAL_PROJECT_TYPE, 
    IDENTIFICATION_GOAL_PRIMARY_LANGUAGE,
    IDENTIFICATION_OBSERVATION_SYSTEM_PROMPT, 
    IDENTIFICATION_PLAN_SYSTEM_PROMPT,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.agents.agent_task import AgentTask

logger = logging.getLogger(__name__)

class IdentificationPlanResult(BaseModel):
    """ Identification Plan Result """
    actions: list[dict] = Field(description="a list of action dictionary, e.g. [{'name': 'read_file', 'input': 'README.md'}, ...]")

IdentificationPlanResultJsonSchema = {
    "title": "identification_plan_result",
    "description": "plan result",
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "description": """a list of action dictionary, e.g. [{'name': 'read_file', 'input': 'README.md'}, ...]""",
            "title": "Actions",
            "items": {"type": "object"}
        },
    },
    "required": ["actions"],
}

class IdentificationState(TypedDict):
    llm: BaseChatOpenAI
    goal: str

    plan: Optional[str]
    plan_reasoning: Optional[str]
    intermediate_steps: Optional[list[str]]
    final_answer: Optional[str]
    step_output: Optional[str]
    step_analysis: Optional[str]
    step_thoughts: Optional[str]

class ProjectTypeEnum(Enum):
    application="application"
    package="package"
    pipeline="pipeline"
    unknown="unknown type"

class PrimaryLanguageEnum(Enum):
    python="python"
    R="R"
    unknown="unknown type"

class IdentificationTask(AgentTask):
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        step_callback: Callable | None=None,
    ):
        super().__init__(llm=llm, step_callback=step_callback)
        self.repo_path: str | None = None
        self.gitignore_path: str | None = None
        self.repo_structure: str | None = None
        self.tools = []
        self.custom_tools = []

    def _initialize(self, repo_path: str, gitignore_path: str):
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path
        if not os.path.exists(self.repo_path):
            raise ValueError(f"Repository path {self.repo_path} does not exist.")
        files = read_directory(self.repo_path, os.path.join(self.repo_path, ".gitignore"))
        file_pairs = [(f, "f" if os.path.isfile(f) else "d") for f in files]
        self.repo_structure = ""
        for f, f_type in file_pairs:
            self.repo_structure += f"{f} - {f_type}\n"

        self.tools = [
            summarize_file_tool(llm=self.llm, repo_path=self.repo_path, token_usage_callback=self._print_step),
            read_directory_tool(repo_path=self.repo_path, gitignore_path=self.gitignore_path),
        ]
        self.custom_tools = [Tool(
            name=tool.__class__.__name__,
            func=tool.run,
            description=tool.__class__.__doc__,
        ) for tool in self.tools]
        self.custom_tools.append(CustomPythonAstREPLTool())
        
    
    def _compile(
        self, 
        repo_path: str,
        gitignore_path: str,
    ):
        self._initialize(repo_path, gitignore_path)
        
        def _initialize_step(state: IdentificationState):
            state["step_output"] = None
            state["step_analysis"] = None
            state["step_thoughts"] = None

        def _build_intermediate_step_and_current_step(state: IdentificationState):
            intermediate_steps = ""
            # previous step
            if "intermediate_steps" in state:
                for i in range(len(state['intermediate_steps'])):
                    step = state['intermediate_steps'][i].replace("{", "_").replace("}", "_")
                    intermediate_steps += step + "\n"
            # current step
            if "step_output" in state and state["step_output"] is not None:
                step_content = state["step_output"]
                step_content = step_content.replace("{", "_").replace("}", "_")
                intermediate_steps += step_content
            return intermediate_steps
        
        def _build_intermediate_analysis_and_thoughts(state: IdentificationState):
            intermediate_analysis = "N/A" if "step_analysis" not in state or \
                state["step_analysis"] is None \
                else state["step_analysis"]
            intermediate_thoughts = "N/A" if "step_thoughts" not in state or \
                state["step_thoughts"] is None \
                else state["step_analysis"]
            return intermediate_analysis, intermediate_thoughts
        
        def _build_plan_prompt(state: IdentificationState):
            goal = state['goal']
            repo_structure = self.repo_structure
            intermediate_steps = _build_intermediate_step_and_current_step(state)
            step_analysis, step_thoughts = _build_intermediate_analysis_and_thoughts(state)
            self._print_step(
                step_output="**Intermediate Step Output**\n" + intermediate_steps
            )
            self._print_step(
                step_output=f"**Intermediate Step Analysis**\n{step_analysis}\n**Intermediate Step Thoughts**\n{step_thoughts}"
            )

            custom_tool_names, custom_tools_desc = get_tool_names_and_descriptions(self.custom_tools)
            return IDENTIFICATION_PLAN_SYSTEM_PROMPT.format(
                goal = goal,
                repo_structure=repo_structure,
                tools=custom_tools_desc,
                intermediate_steps=intermediate_steps,
                tool_names=custom_tool_names,
                intermediate_analysis=step_analysis,
                intermediate_thoughts=step_thoughts,
            )
        def _build_observation_prompt(state: IdentificationState):
            goal = state["goal"]
            repo_structure = self.repo_structure
            intermediate_steps = _build_intermediate_step_and_current_step(state)
            prompt = ChatPromptTemplate.from_template(IDENTIFICATION_OBSERVATION_SYSTEM_PROMPT)

            return prompt.format(
                goal=goal,
                repo_structure=repo_structure,
                intermediate_output=intermediate_steps,
            )
        def _convert_plan(plan: IdentificationPlanResult):
            result_plan = ""
            for action in plan.actions:
                result_plan += f"Step: {action['name']}\nStep Input: {action['input']}\n"
            
            return result_plan
        def plan_step(state: IdentificationState):
            self._print_step(step_name="Plan Step")
            syste_prompt = _build_plan_prompt(state)
            agent = CommonAgentTwoSteps(llm=self.llm)
            res, processed_res, token_useage, reasoning_process = agent.go(
                system_prompt=syste_prompt,
                instruction_prompt="Now, let's begin",
                schema=IdentificationPlanResultJsonSchema,
            )

            _initialize_step(state)            
            res = IdentificationPlanResult(**res)
            self._print_step(step_output=f"**Reasoning Process**\n{reasoning_process}")
            self._print_step(step_output=f"**Plan**\n{str(res.actions)}")
            self._print_step(token_usage=token_useage)
            state['plan'] = _convert_plan(res)
            return state

        def execute_step(state: IdentificationState):
            self._print_step(step_name="Execute Step")
            plan = state['plan']
            prompt = CustomPromptTemplate(
                template=IDENTIFICATION_EXECUTION_SYSTEM_PROMPT,
                tools=self.custom_tools,
                plan_actions=plan,
                input_variables=["tools", "agent_scratchpad", "intermediate_steps", "tool_names", "plan_actions"]
            )
            output_parser = CustomOutputParser()
            agent = create_react_agent(
                llm=self.llm,
                tools=self.custom_tools,
                prompt=prompt,
                output_parser=output_parser,
                stop_sequence=["\nObservation:"],
            )
            callback_handler = OpenAICallbackHandler()
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=self.custom_tools
            )
            response = agent_executor.invoke(
                input={"plan_actions": plan, "input": "Now, let's begin."},
                config={"callbacks": [callback_handler]}
            )
            
            # parse response
            if 'output' in response:
                output = response["output"]
                if "**Final Answer**" in output:
                    pattern = r"\*\*Final Answer\*\*\s*\d*\s*(.*)"
                    match = re.search(pattern, output, re.DOTALL)
                    if not match:
                        logger.error("Can't find 'Final Answer' in execution output")
                        state["step_output"] = ""
                        return state
                    step_output = match.group(1).strip().strip(':')
                    self._print_step(step_output=step_output)
                    self._print_step(token_usage=callback_handler)
                    state["step_output"] = step_output
                    return state
                elif "Final Answer" in output:
                    pattern = r"Final Answer\s*\d*\s*(.*)"
                    match = re.search(pattern, output, re.DOTALL)
                    if not match:
                        logger.error("Can't find 'Final Answer' in execution output")
                        state["step_output"] = ""
                        return state
                    step_output = match.group(1).strip().strip(':')
                    self._print_step(step_output=step_output)
                    self._print_step(token_usage=callback_handler)
                    state["step_output"] = step_output
                    return state
                else:
                    state["step_output"] = output

            return state

        def observe_step(state: IdentificationState):
            self._print_step(step_name="Observe Step")
            intermediate_steps = state["intermediate_steps"] if "intermediate_steps" in state else []
            agent = CommonAgentTwoSteps(llm=self.llm)
            res, processed_res, token_usage, reasoning_process = agent.go(
                system_prompt=_build_observation_prompt(state),
                instruction_prompt="Let's begin thinking.",
                schema=ObservationResult
            )
            res: ObservationResult = res
            state["final_answer"] = res.FinalAnswer
            analysis = res.Analysis
            thoughts = res.Thoughts
            self._print_step(step_output=f"**Observation Reasoning Process**\n{reasoning_process}")
            self._print_step(
                step_output=f"Final Answer: {res.FinalAnswer if res.FinalAnswer else None}\nAnalysis: {analysis}\nThoughts: {thoughts}",
            )
            self._print_step(token_usage=token_usage)
            
            intermediate_step: list[str] = state["intermediate_steps"] if "intermediate_steps" in state else []
            step_output = state["step_output"]
            intermediate_step.append(step_output)
            state["intermediate_steps"] = intermediate_step
            state["step_analysis"] = analysis
            state["step_thoughts"] = thoughts

            print(state)
            return state
        
        def check_observation_step(state: IdentificationState):
            if "final_answer" in state and state["final_answer"] is not None:
                return END
            return "plan_step"
        
        graph = StateGraph(IdentificationState)
        graph.add_node("plan_step", plan_step)
        graph.add_node("execute_step", execute_step)
        graph.add_node("observe_step", observe_step)
        graph.add_edge(START, "plan_step")
        graph.add_edge("plan_step", "execute_step")
        graph.add_edge("execute_step", "observe_step")
        graph.add_conditional_edges("observe_step", check_observation_step, {"plan_step", END})

        self.graph = graph.compile()
        
    def identify_project_type(self):
        s = self._go_graph({"goal": IDENTIFICATION_GOAL_PROJECT_TYPE})
        proj_type = s["final_answer"] if "final_answer" in s else "unknown type"
        return self._parse_project_type(proj_type)
    
    def identify_primary_language(self):
        s = self._go_graph({"goal": IDENTIFICATION_GOAL_PRIMARY_LANGUAGE})
        language = s["final_answer"] if "final_answer" in s else "unknown type"
        return self._parse_primary_language(language)
    
    def _parse_project_type(self, proj_type: str) -> ProjectTypeEnum:
        proj_type = proj_type.strip()
        if proj_type == "application":
            return ProjectTypeEnum.application
        elif proj_type == "package":
            return ProjectTypeEnum.package
        elif proj_type == "pipeline":
            return ProjectTypeEnum.pipeline
        else:
            return ProjectTypeEnum.unknown
        
    def _parse_primary_language(self, language: str) -> PrimaryLanguageEnum:
        language = language.strip()
        if language == "python":
            return PrimaryLanguageEnum.python
        elif language == "R":
            return PrimaryLanguageEnum.R
        else:
            return PrimaryLanguageEnum.unknown
