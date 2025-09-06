import os
from typing import Callable, Optional, TypedDict
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import AIMessage
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import logging
from langgraph.graph import StateGraph, START, END

from bioguider.agents.agent_task import AgentTask
from bioguider.agents.agent_tools import agent_tool
from bioguider.agents.agent_utils import read_file, summarize_file
from bioguider.agents.peo_common_step import PEOWorkflowState
from bioguider.agents.common_agent import CommonAgent
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.consistency_collection_task_utils import (
    ConsistencyCollectionWorkflowState,
    retrieve_function_definition_and_docstring_tool,
    retrieve_class_definition_and_docstring_tool,
    retrieve_class_and_method_definition_and_docstring_tool,
    retrieve_method_definition_and_docstring_tool,
)
from bioguider.agents.consistency_collection_plan_step import ConsistencyCollectionPlanStep
from bioguider.agents.consistency_collection_observe_step import ConsistencyCollectionObserveStep
from bioguider.agents.consistency_collection_execute_step import ConsistencyCollectionExecuteStep
from bioguider.database.code_structure_db import CodeStructureDb

logger = logging.getLogger(__name__)

class ConsistencyCollectionTask(AgentTask):
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        code_structure_db: CodeStructureDb,
        step_callback: Callable | None = None,
    ):
        super().__init__(llm=llm, step_callback=step_callback)
        self.llm = llm
        self.code_structure_db = code_structure_db
        
        func_tool = retrieve_function_definition_and_docstring_tool(llm=llm, code_structure_db=code_structure_db)
        class_tool = retrieve_class_definition_and_docstring_tool(llm=llm, code_structure_db=code_structure_db)
        class_and_method_tool = retrieve_class_and_method_definition_and_docstring_tool(llm=llm, code_structure_db=code_structure_db)
        method_tool = retrieve_method_definition_and_docstring_tool(llm=llm, code_structure_db=code_structure_db)
        self.tools = [func_tool, class_tool, class_and_method_tool, method_tool]
        self.custom_tools = [            
            StructuredTool.from_function(
                func_tool.run,
                description=func_tool.__class__.__doc__,
                name=func_tool.__class__.__name__,
            ),            
            StructuredTool.from_function(
                class_tool.run,
                description=class_tool.__class__.__doc__,
                name=class_tool.__class__.__name__,
            ),            
            StructuredTool.from_function(
                class_and_method_tool.run,
                description=class_and_method_tool.__class__.__doc__,
                name=class_and_method_tool.__class__.__name__,
            ),
            StructuredTool.from_function(
                method_tool.run,
                description=method_tool.__class__.__doc__,
                name=method_tool.__class__.__name__,
            ),
        ]

        self.steps = [
            ConsistencyCollectionPlanStep(llm=llm, custom_tools=self.custom_tools), 
            ConsistencyCollectionExecuteStep(llm=llm, code_structure_db=code_structure_db, custom_tools=self.custom_tools),
            ConsistencyCollectionObserveStep(llm=llm)
        ]

    def _compile(self, repo_path: str, gitignore_path: str):
        def check_observe_step(state: ConsistencyCollectionWorkflowState):
            if "final_answer" in state and state["final_answer"] is not None:
                return END
            return "plan_step"
        def check_plan_step(state: ConsistencyCollectionWorkflowState):
            if "plan_actions" in state and state["plan_actions"] is not None and len(state["plan_actions"]) > 0:
                return "execute_step"
            return END
        
        graph = StateGraph(ConsistencyCollectionWorkflowState)
        graph.add_node("plan_step", self.steps[0].execute)
        graph.add_node("execute_step", self.steps[1].execute)
        graph.add_node("observe_step", self.steps[2].execute)
        graph.add_edge(START, "plan_step")
        graph.add_conditional_edges("plan_step", check_plan_step, {"execute_step", END})
        graph.add_edge("execute_step", "observe_step")
        graph.add_conditional_edges("observe_step", check_observe_step, {"plan_step", END})

        self.graph = graph.compile()

    def collect(self, user_guide_api_documentation: str) -> tuple[bool, str | None]:
        s = self._go_graph({
            "user_guide_api_documentation": user_guide_api_documentation,
            "step_count": 0,
        })
        # analyze the final assembly result
        if "final_assembly_result" in s and s["final_assembly_result"] is not None:
            self._print_step(step_name="Final Assembly Result")
            self._print_step(step_output=s["final_assembly_result"])
            return True, s["final_assembly_result"]
        else:
            return False, s["thoughts"] if "thoughts" in s else None
        