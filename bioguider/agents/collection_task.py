
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

from bioguider.agents.common_agent import CommonAgent
from bioguider.agents.agent_tools import read_directory_tool, summarize_file_tool, read_file_tool
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.agents.agent_task import AgentTask

class CollectionTask(AgentTask):
    def __init__(self, llm: BaseChatOpenAI, step_callback: Callable | None = None):
        super().__init__(llm, step_callback)
        self._repo_path: str | None = None
        self._gitignore_path: str | None = None
    
    def _initialize(self):
        self.tools = [
            read_directory_tool(repo_path=self._repo_path),
            summarize_file_tool(
                llm=self._llm,
                repo_path=self._repo_path,
                token_usage_callback=self._token_usage_callback,
            ),
            read_file_tool(repo_path=self._repo_path),
        ]
        self.custom_tools = [Tool(
            name=tool.__class__.__name__,
            func=tool.run,
            description=tool.__class__.__doc__,
        ) for tool in self.tools]
        self.custom_tools.append(CustomPythonAstREPLTool())

    def _compile(self, repo_path, gitignore_path):
        self._repo_path = repo_path
        self._gitignore_path = gitignore_path
        self._initialize()

        

            




