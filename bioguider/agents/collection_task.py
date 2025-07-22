
import os
import logging
import re
import json
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
from bioguider.utils.file_utils import get_file_type
from bioguider.agents.agent_utils import read_directory, try_parse_json_object
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

logger = logging.getLogger(__name__)

class CollectionTask(AgentTask):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        step_callback: Callable | None = None,
        summarize_instruction: str | None = "N/A",
        summarized_files_db: SummarizedFilesDb | None = None,
        provided_files: list[str] | None = None,
    ):
        super().__init__(llm, step_callback, summarized_files_db=summarized_files_db)
        self.repo_path: str | None = None
        self.gitignore_path: str | None = None
        self.repo_structure: str | None = None
        self.goal_item: str | None = None
        self.steps: list[PEOCommonStep] = []
        self.tools: list[any] | None = None
        self.custom_tools: list[Tool] | None = None
        self.summarize_instruction = summarize_instruction
        self.provided_files = provided_files

    def _prepare_tools(self, related_file_goal_item_desc):
        tool_rd = read_directory_tool(repo_path=self.repo_path)
        tool_sum = summarize_file_tool(
            llm=self.llm,
            repo_path=self.repo_path,
            output_callback=self.step_callback,
            db=self.summarized_files_db,
            summaize_instruction=self.summarize_instruction,
        )
        tool_rf = read_file_tool(repo_path=self.repo_path)
        tool_cf = check_file_related_tool(
            llm=self.llm,
            repo_path=self.repo_path,
            goal_item_desc=related_file_goal_item_desc,
            output_callback=self.step_callback,
            summarize_instruction=self.summarize_instruction,
            summarized_files_db=self.summarized_files_db,
        )
        self.tools = [tool_rd, tool_sum, tool_rf, tool_cf]
        self.custom_tools = [
            Tool(
                name = tool_rd.__class__.__name__,
                func = tool_rd.run,
                description=tool_rd.__class__.__doc__,
            ),
            StructuredTool.from_function(
                tool_sum.run,
                description=tool_sum.__class__.__doc__,
                name=tool_sum.__class__.__name__,
            ),
            Tool(
                name = tool_rf.__class__.__name__,
                func = tool_rf.run,
                description=tool_rf.__class__.__doc__,
            ),
            Tool(
                name = tool_cf.__class__.__name__,
                func = tool_cf.run,
                description=tool_cf.__class__.__doc__,
            ),
        ]
        # self.custom_tools.append(CustomPythonAstREPLTool())

    def _initialize(self):
        # initialize the 2-level file structure of the repo
        if not os.path.exists(self.repo_path):
            raise ValueError(f"Repository path {self.repo_path} does not exist.")
        files = self.provided_files
        if files is None:
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
        
        self._prepare_tools(related_file_goal_item_desc)
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

    def collect(self) -> list[str] | None:
        s = self._go_graph({"goal_item": self.goal_item})
        if s is None or 'final_answer' not in s:
            return None
        if s["final_answer"] is None:
            return None
        result = s["final_answer"].strip()
        the_obj = try_parse_json_object(result)
        if the_obj is None or "final_answer" not in the_obj:
            logger.error(f"Final answer is not a valid JSON: {result}")
            return None
        final_result = the_obj["final_answer"]
        if isinstance(final_result, str):
            final_result = final_result.strip()
            return [final_result]
        elif isinstance(final_result, list):
            return final_result
        else:
            logger.error(f"Final answer is not a valid JSON list or string: {result}")
            return None
