
import os
import json
from json import JSONDecodeError
import re
import logging
from typing import Callable, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import tool, Tool, BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from bioguider.agents.agent_utils import DEFAULT_TOKEN_USAGE, CustomOutputParser, CustomPromptTemplate, increase_token_usage, read_directory, read_file, summarize_file
from bioguider.agents.common_agent import CommonAgent
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.prompt_utils import (
    IDENTIFICATION_EXECUTION_SYSTEM_PROMPT, 
    IDENTIFICATION_GOAL_PROJECT_TYPE, 
    IDENTIFICATION_OBSERVATION_SYSTEM_PROMPT, 
    IDENTIFICATION_PLAN_SYSTEM_PROMPT,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool

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

class IdentificationObservationResult(BaseModel):
    Analysis: Optional[str]=Field(description="Analyzing the goal, repository file structure and intermediate output.")
    FinalAnswer: Optional[str]=Field(description="the final answer for the goal")
    Thoughts: Optional[str]=Field(description="If the information is insufficient, the thoughts will be given and be taken into consideration in next round.")

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

class IdentificationStep:
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        step_callback: Callable | None=None,
    ):
        self.llm = llm
        self.repo_path: str | None = None
        self.gitignore_path: str | None = None
        self.repo_structure: str | None = None
        self.step_callback: Callable | None = step_callback
    
    def compile(
        self, 
        repo_path: str,
        gitignore_path: str,
    ):
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path

        def _print_step(
            step_name: str | None = None,
            step_output: str | None = None,
            token_usage: dict | object | None = None,
        ):
            if self.step_callback is None:
                return
            # convert token_usage to dict
            if token_usage is not None and not isinstance(token_usage, dict):
                token_usage = vars(token_usage)
            step_callback = self.step_callback
            step_callback(
                step_name=step_name,
                step_output=step_output,
                token_usage=token_usage,
            )

        @tool
        def analyze_file_tool(file_path: str) -> str | None:
            """ read and analyze file
Args:
    file_path str: file path
Returns:
    A string of analysis result, if the file does not exist, return None. 
    The fille will be analyzed to infer the project type.
            """
            if file_path is None:
                return None
            file_path = file_path.strip()
            if repo_path not in file_path:
                file_path = os.path.join(self.repo_path, file_path)
            content = read_file(file_path)
            if content is None:
                return None
            summarized_content, token_usage = summarize_file(
                self.llm, 
                file_path, 
                content, 
                IDENTIFICATION_GOAL_PROJECT_TYPE, 
                level=6
            )
            _print_step(token_usage=token_usage)
            return summarized_content
        
        @tool
        def read_directory_tool(dir_path: str) -> str | None:
            """ 
Reads the contents of a directory, including files and subdirectories up to the specified depth.
This function will exclude the file or directories specified in .gitignore file (gitignore_path).
Args:
    dir_path (str): Path to the directory.
Returns:
    a string containing file and subdirectory paths found within the specified depth.
           """
            full_path = dir_path
            if full_path == "." or full_path == "..":
                return f"Please skip this folder {dir_path}"
            if repo_path not in full_path:
                full_path = os.path.join(repo_path, full_path)
            files = read_directory(full_path, gitignore_path=gitignore_path, level=1)
            if files is None:
                return "N/A"
            file_pairs = [(f, "f" if os.path.isfile(os.path.join(full_path, f)) else "d") for f in files]
            dir_structure = ""
            for f, f_type in file_pairs:
                dir_structure += f"{os.path.join(dir_path, f)} - {f_type}\n"
            return dir_structure
        
        python_tool = Tool(
            name="Custom_Python_AST_REPL",
            func=CustomPythonAstREPLTool().run,
            description="Executes Python code. Useful for math, logic, or small computations."
        )
        custom_tools = [python_tool, analyze_file_tool, read_directory_tool]
        custom_tools_desc = ""
        custom_tool_names = "["
        for t in custom_tools:
            custom_tools_desc += f"name: {t.name}, description: {t.description}\n"
            custom_tool_names += t.name + ","
        custom_tool_names += "]"

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
            _print_step(
                step_output="**Intermediate Step Output**\n" + intermediate_steps
            )
            _print_step(
                step_output=f"**Intermediate Step Analysis**\n{step_analysis}\n**Intermediate Step Thoughts**\n{step_thoughts}"
            )

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
            _print_step(step_name="Plan Step")
            syste_prompt = _build_plan_prompt(state)
            agent = CommonAgentTwoSteps(llm=self.llm)
            res, processed_res, token_useage, reasoning_process = agent.go(
                system_prompt=syste_prompt,
                instruction_prompt="Now, let's begin",
                schema=IdentificationPlanResultJsonSchema,
            )

            _initialize_step(state)            
            res = IdentificationPlanResult(**res)
            _print_step(step_output=f"**Reasoning Process**\n{reasoning_process}")
            _print_step(step_output=f"**Plan**\n{str(res.actions)}")
            _print_step(token_usage=token_useage)
            state['plan'] = _convert_plan(res)
            return state

        def execute_step(state: IdentificationState):
            _print_step(step_name="Execute Step")
            plan = state['plan']
            prompt = CustomPromptTemplate(
                template=IDENTIFICATION_EXECUTION_SYSTEM_PROMPT,
                tools=custom_tools,
                plan=plan,
                input_variables=["tools", "agent_scratchpad", "intermediate_steps", "tool_names", "plan_steps"]
            )
            output_parser = CustomOutputParser()
            agent = create_react_agent(
                llm=self.llm,
                tools=custom_tools,
                prompt=prompt,
                output_parser=output_parser,
                stop_sequence=["\nObservation:"],
            )
            callback_handler = OpenAICallbackHandler()
            agent_executor = AgentExecutor(agent=agent, tools=custom_tools)
            response = agent_executor.invoke(
                input={"plan_steps": plan, "input": "Now, let's begin."},
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
                    _print_step(step_output=step_output)
                    _print_step(token_usage=callback_handler)
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
                    _print_step(step_output=step_output)
                    _print_step(token_usage=callback_handler)
                    state["step_output"] = step_output
                    return state
                else:
                    state["step_output"] = output

            return state

        def observe_step(state: IdentificationState):
            _print_step(step_name="Observe Step")
            intermediate_steps = state["intermediate_steps"] if "intermediate_steps" in state else []
            agent = CommonAgentTwoSteps(llm=self.llm)
            res, processed_res, token_usage, reasoning_process = agent.go(
                system_prompt=_build_observation_prompt(state),
                instruction_prompt="Let's begin thinking.",
                schema=IdentificationObservationResult
            )
            res: IdentificationObservationResult = res
            state["final_answer"] = res.FinalAnswer
            analysis = res.Analysis
            thoughts = res.Thoughts
            _print_step(step_output=f"**Observation Reasoning Process**\n{reasoning_process}")
            _print_step(
                step_output=f"Final Answer: {res.FinalAnswer if res.FinalAnswer else None}\nAnalysis: {analysis}\nThoughts: {thoughts}",
            )
            _print_step(token_usage=token_usage)
            
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
        
    def execute(self):
        files = read_directory(self.repo_path, os.path.join(self.repo_path, ".gitignore"))
        file_pairs = [(f, "f" if os.path.isfile(f) else "d") for f in files]
        self.repo_structure = ""
        for f, f_type in file_pairs:
            self.repo_structure += f"{f} - {f_type}\n"
        
        for s in self.graph.stream(input={
            "llm": self.llm,
            "goal": IDENTIFICATION_GOAL_PROJECT_TYPE,
        }, stream_mode="values"):
            print(s)

        return s["final_answer"] if "final_answer" in s else "unknown type"
