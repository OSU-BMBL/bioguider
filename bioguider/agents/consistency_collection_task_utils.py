import os
from pathlib import Path
from typing import Callable, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
import logging

from bioguider.agents.agent_tools import agent_tool
from bioguider.database.code_structure_db import CodeStructureDb

logger = logging.getLogger(__name__)

class ConsistencyCollectionWorkflowState(TypedDict):
    user_guide_api_documentation: str
    functions_and_classes: Optional[list[dict]]
    all_query_rows: Optional[list[any]]
    step_output_callback: Optional[Callable]
    intermediate_steps: Optional[str]
    step_output: Optional[str]
    step_analysis: Optional[str]
    step_thoughts: Optional[str]
    plan_actions: Optional[list[dict]]

    final_answer: Optional[str]
    final_assembly_result: Optional[str]
    step_count: Optional[int]

class ConsistencyEvaluationState(TypedDict):
    user_guide_api_documentation: str
    step_output_callback: Optional[Callable]
    functions_and_classes: Optional[list[dict]]
    all_query_rows: Optional[list[any]]
    consistency_score: Optional[str]
    consistency_assessment: Optional[str]
    consistency_development: Optional[list[str]]
    consistency_strengths: Optional[list[str]]

class retrieve_method_definition_and_docstring_tool:
    """ Retrieve the method definition and docstring.
    If the method is a method of a class, you **must** put the class name as the parent name and better to put the file path as the file path of the class.
Args:
    method_name str: the name of the method
    class_name str: the name of the class that the method is in.
    file_path str: the path of the file that the method is in. If not sure, just put "N/A"
Returns:
    str: the method definition and docstring
    """
    def __init__(self, llm: BaseChatOpenAI, code_structure_db: CodeStructureDb):
        self.llm = llm
        self.code_structure_db = code_structure_db

    def run(self, method_name: str, class_name: str, file_path: str) -> str:
        if file_path != "N/A":
            row = self.code_structure_db.select_by_name_and_parent_and_path(method_name, class_name, file_path)
            if row is None:
                return "Can't retrieve method definition and docstring"
            return f"Method: {row['name']}\nDocstring: {row['doc_string']}\nParams: {row['params']}"
        else:
            rows = self.code_structure_db.select_by_name_and_parent(method_name, class_name)
            if rows is None or len(rows) == 0:
                return "Can't retrieve method definition and docstring"
            return f"Method: {rows[0]['name']}\nDocstring: {rows[0]['doc_string']}\nParams: {rows[0]['params']}"

class retrieve_function_definition_and_docstring_tool:
    """ Retrieve the function definition and docstring
Args:
    function_name str: the name of the function
    file_path str: the path of the file that the function is in. If not sure, just put "N/A"
Returns:
    str: the function definition and docstring
    """
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        code_structure_db: CodeStructureDb,
    ):
        self.llm = llm
        self.code_structure_db = code_structure_db

    def run(self, function_name: str, file_path: str) -> str:
        if file_path != "N/A":
            row = self.code_structure_db.select_by_name_and_path(function_name, file_path)
            if row is None:
                return f"No such function {function_name}"
            return f"Function: {row['name']}\nDocstring: {row['doc_string']}\nParams: {row['params']}"
        else:
            rows = self.code_structure_db.select_by_name(function_name)
            if rows is None or len(rows) == 0:
                return f"No such function {function_name}"
            return f"Function: {rows[0]['name']}\nDocstring: {rows[0]['doc_string']}\nParams: {rows[0]['params']}"

class retrieve_class_definition_and_docstring_tool:
    """ Retrieve the class definition and docstring
Args:
    class_name str: the name of the class
    file_path str: the path of the file that the class is in. If not sure, just put "N/A"
Returns:
    str: the class definition and docstring
    """
    def __init__(self, llm: BaseChatOpenAI, code_structure_db: CodeStructureDb):
        self.llm = llm
        self.code_structure_db = code_structure_db

    def run(self, class_name: str, file_path: str) -> str:
        if file_path != "N/A":
            row = self.code_structure_db.select_by_name_and_path(class_name, file_path)
            if row is None:
                return f"No such class {class_name}"
            return f"Class: {row['name']}\nDocstring: {row['doc_string']}\nParams: {row['params']}"
        else:
            rows = self.code_structure_db.select_by_name(class_name)
            if rows is None or len(rows) == 0:
                return f"No such class {class_name}"
            return f"Class: {rows[0]['name']}\nDocstring: {rows[0]['doc_string']}\nParams: {rows[0]['params']}"

class retrieve_class_and_method_definition_and_docstring_tool:
    """ Retrieve the class and all methods definition and docstring
Args:
    class_name str: the name of the class
    file_path str: the path of the file that the class is in. If not sure, just put "N/A"
Returns:
    str: the class and method definition and docstring
    """
    def __init__(self, llm: BaseChatOpenAI, code_structure_db: CodeStructureDb):
        self.llm = llm
        self.code_structure_db = code_structure_db

    def run(self, class_name: str, file_path: str) -> str:
        if file_path != "N/A":
            row = self.code_structure_db.select_by_name_and_path(class_name, file_path)
            if row is None:
                return f"No such class {class_name}"
        else:
            rows = self.code_structure_db.select_by_name(class_name)
            if rows is None or len(rows) == 0:
                return f"No such class {class_name}"
            row = rows[0]
        
        parent_path = file_path if file_path is not None and file_path.lower() != "n/a" else row["path"]
        methods = self.code_structure_db.select_by_parent(
            class_name, 
            parent_path
        )
        method_definitions = []
        for method in methods:
            method_definitions.append(f"Method: {method['name']}\nDocstring: {method['doc_string']}\nParams: {method['params']}\n\n")
        return f"Class: {row['name']}\nDocstring: {row['doc_string']}\nParams: {row['params']}\nMethods: {method_definitions}"