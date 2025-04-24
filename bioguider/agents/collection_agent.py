
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
from bioguider.agents.agent_tools import read_directory, summarize_file, read_file
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.agents.agent_step import AgentStep

class CollectionStep(AgentStep):
    def __init__(self, llm: BaseChatOpenAI, step_callback: Callable | None = None):
        super()().__init__(llm, step_callback)

    # def 
            

        




