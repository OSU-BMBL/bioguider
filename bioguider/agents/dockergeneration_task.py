
import os
import re
from pydantic import BaseModel, Field
from typing import Callable, List, Optional, TypedDict, Union
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import StructuredTool, Tool, tool, BaseTool
from langchain.agents import (
    initialize_agent, 
    AgentType, 
    AgentOutputParser,
    create_react_agent,
    AgentExecutor,
)
from langchain.schema import (
    AgentFinish,
    AgentAction,
)
from langgraph.graph import StateGraph, START, END

from bioguider.database.summarized_file_db import SummarizedFilesDb
from bioguider.agents.peo_common_step import PEOCommonStep
from bioguider.utils.file_utils import get_file_type
from bioguider.agents.agent_utils import read_directory, read_file
from bioguider.agents.collection_task_utils import (
    RELATED_FILE_GOAL_ITEM,
    CollectionWorkflowState, 
    check_file_related_tool,
)
from bioguider.agents.common_agent import CommonAgent
from bioguider.agents.dockergeneration_task_utils import (
    generate_Dockerfile_tool,
    prepare_provided_files_string,
    write_file_tool,
    extract_python_file_from_notebook_tool,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.agents.dockergeneration_plan_step import DockerGenerationPlanStep
from bioguider.agents.dockergeneration_execute_step import DockerGenerationExecuteStep
from bioguider.agents.dockergeneration_observe_step import DockerGenerationObserveStep
from bioguider.agents.dockergeneration_task_utils import DockerGenerationWorkflowState
from bioguider.agents.agent_task import AgentTask

class DockerGenerationTask(AgentTask):
    def __init__(
        self, 
        llm, 
        step_callback = None,
    ):
        super().__init__(llm, step_callback)
        self.repo_path: str | None = None
        self.gitignore_path: str | None = None
        self.repo_structure: str | None = None
        self.steps: list[PEOCommonStep] = []
        self.tools: list[any] | None = None
        self.provided_files: list[str] | None = None

    def _initialize(self):
        # initialize the 2-level file structure of the repo
        if not os.path.exists(self.repo_path):
            raise ValueError(f"Repository path {self.repo_path} does not exist.")
        files = read_directory(self.repo_path, os.path.join(self.repo_path, ".gitignore"))
        file_pairs = [(f, get_file_type(os.path.join(self.repo_path, f)).value) for f in files]
        self.repo_structure = ""
        for f, f_type in file_pairs:
            self.repo_structure += f"{f} - {f_type}\n"

        # initialize extracted files string
        if self.provided_files is not None:
            self.str_extracted_files = prepare_provided_files_string(
                self.repo_path, self.provided_files
            )
        write_tool = write_file_tool(self.repo_path)
        generate_tool = generate_Dockerfile_tool(
            llm=self.llm,
            repo_path=self.repo_path,
            extracted_files=self.str_extracted_files,
            repo_structure=self.repo_structure,
            output_callback=self.step_callback,
        )
        extract_tool = extract_python_file_from_notebook_tool(
            repo_path=self.repo_path,
        )
        self.tools = [
            write_tool, generate_tool, extract_tool,
        ]
        self.custom_tools = [
            StructuredTool.from_function(
                write_tool.run,
                description=write_tool.__class__.__doc__,
                name=write_tool.__class__.__name__,
            ),
            Tool(
                func=generate_tool.run,
                description=generate_tool.__class__.__doc__,
                name=generate_tool.__class__.__name__,
            ),
            StructuredTool.from_function(
                extract_tool.run,
                description=extract_tool.__class__.__doc__,
                name=extract_tool.__class__.__name__,
            )
        ]
        self.custom_tools.append(CustomPythonAstREPLTool())
        plan_step = DockerGenerationPlanStep(
            llm=self.llm,
            repo_path=self.repo_path,
            repo_structure=self.repo_structure,
            gitignore_path=self.gitignore_path,
            custom_tools=self.custom_tools,
        )
        execute_step = DockerGenerationExecuteStep(
            llm=self.llm,
            repo_path=self.repo_path,
            repo_structure=self.repo_structure,
            gitignore_path=self.gitignore_path,
            custom_tools=self.custom_tools,
        )
        observe_step = DockerGenerationObserveStep(
            llm=self.llm,
            repo_path=self.repo_path,
        )
        self.steps = [
            plan_step, execute_step, observe_step,
        ]
        # pass generate_Dockerfile_tool to execute step
        execute_step.set_generate_Dockerfile_tool(generate_tool)

    def _compile(self, repo_path, gitignore_path, **kwargs):
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path
        self.provided_files = kwargs.get("provided_files")
        self._initialize()

        def check_observe_step(state: DockerGenerationWorkflowState):
            if "final_answer" in state and state["final_answer"] is not None:
                self._print_step(step_name="Final Answer")
                self._print_step(step_output=state["final_answer"])
                return END
            return "plan_step"
        
        graph = StateGraph(DockerGenerationWorkflowState)
        graph.add_node("plan_step", self.steps[0].execute)
        graph.add_node("execute_step", self.steps[1].execute)
        graph.add_node("observe_step", self.steps[2].execute)
        graph.add_edge(START, "plan_step")
        graph.add_edge("plan_step", "execute_step")
        graph.add_edge("execute_step", "observe_step")
        graph.add_conditional_edges("observe_step", check_observe_step, {"plan_step", END})

        self.graph = graph.compile()

    def generate(self):
        s = self._go_graph({"provided_files": self.provided_files})
        return s

