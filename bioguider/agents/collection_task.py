
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

from bioguider.utils.file_utils import get_file_type
from bioguider.agents.agent_utils import read_directory
from bioguider.agents.collection_task_utils import (
    RELATED_FILE_GOAL_ITEM,
    CollectionWorkflowState, 
    check_file_related_tool,
)
from bioguider.agents.common_agent import CommonAgent
from bioguider.agents.agent_tools import (
    read_directory_tool, 
    summarize_file_tool, 
    read_file_tool,
)
from bioguider.agents.peo_common_step import PEOCommonStep
from bioguider.agents.prompt_utils import COLLECTION_PROMPTS
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.agents.agent_task import AgentTask
from bioguider.agents.collection_plan_step import CollectionPlanStep
from bioguider.agents.collection_execute_step import CollectionExecuteStep
from bioguider.agents.collection_observe_step import CollectionObserveStep

class CollectionTask(AgentTask):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        step_callback: Callable | None = None
    ):
        super().__init__(llm, step_callback)
        self.repo_path: str | None = None
        self.gitignore_path: str | None = None
        self.repo_structure: str | None = None
        self.goal_item: str | None = None
        self.steps: list[PEOCommonStep] = []
        self.tools: list[any] | None = None
        self.custom_tools: list[Tool] | None = None

    def _initialize(self):
        # initialize the 2-level file structure of the repo
        if not os.path.exists(self.repo_path):
            raise ValueError(f"Repository path {self.repo_path} does not exist.")
        files = read_directory(self.repo_path, os.path.join(self.repo_path, ".gitignore"))
        file_pairs = [(f, get_file_type(os.path.join(self.repo_path, f)).value) for f in files]
        self.repo_structure = ""
        for f, f_type in file_pairs:
            self.repo_structure += f"{f} - {f_type}\n"
            
        collection_item = COLLECTION_PROMPTS[self.goal_item]
        related_file_goal_item_desc = ChatPromptTemplate.from_template(RELATED_FILE_GOAL_ITEM).format(
            goal_item=collection_item["goal_item"],
            related_file_description=collection_item["related_file_description"],
        )
        self.tools = [
            read_directory_tool(repo_path=self.repo_path),
            summarize_file_tool(
                llm=self.llm,
                repo_path=self.repo_path,
                output_callback=self.step_callback,
            ),
            read_file_tool(repo_path=self.repo_path),
            check_file_related_tool(
                llm=self.llm,
                repo_path=self.repo_path,
                goal_item_desc=related_file_goal_item_desc,
                output_callback=self.step_callback,
            ),
        ]
        self.custom_tools = [Tool(
            name=tool.__class__.__name__,
            func=tool.run,
            description=tool.__class__.__doc__,
        ) for tool in self.tools]
        self.custom_tools.append(CustomPythonAstREPLTool())
        self.steps = [
            CollectionPlanStep(
                llm=self.llm,
                repo_path=self.repo_path,
                repo_structure=self.repo_structure,
                gitignore_path=self.gitignore_path,
                custom_tools=self.custom_tools,
            ),
            CollectionExecuteStep(
                llm=self.llm,
                repo_path=self.repo_path,
                repo_structure=self.repo_structure,
                gitignore_path=self.gitignore_path,
                custom_tools=self.custom_tools,
            ),
            CollectionObserveStep(
                llm=self.llm,
                repo_path=self.repo_path,
                repo_structure=self.repo_structure,
                gitignore_path=self.gitignore_path,
            ),
        ]

    def _compile(self, repo_path: str, gitignore_path: str, **kwargs):
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path
        self.goal_item = kwargs.get("goal_item")
        self._initialize()

        def check_observe_step(state):
            if "final_answer" in state and state["final_answer"] is not None:
                self._print_step(step_name="Final Answer")
                self._print_step(step_output=state["final_answer"])
                return END
            return "plan_step"

        graph = StateGraph(CollectionWorkflowState)
        graph.add_node("plan_step", self.steps[0].execute)
        graph.add_node("execute_step", self.steps[1].execute)
        graph.add_node("observe_step", self.steps[2].execute)
        graph.add_edge(START, "plan_step")
        graph.add_edge("plan_step", "execute_step")
        graph.add_edge("execute_step", "observe_step")
        graph.add_conditional_edges("observe_step", check_observe_step, {"plan_step", END})

        self.graph = graph.compile()

    def collect(self):
        s = self._go_graph({"goal_item": self.goal_item})
        return s

        

            




