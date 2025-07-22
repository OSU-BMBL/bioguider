
from typing import Callable
from abc import ABC, abstractmethod

from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph.graph import CompiledGraph

from bioguider.utils.constants import DEFAULT_TOKEN_USAGE
from bioguider.database.summarized_file_db import SummarizedFilesDb

class AgentTask(ABC):
    """
    A class representing a step in an agent's process.
    """

    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        step_callback: Callable | None = None,
        summarized_files_db: SummarizedFilesDb | None = None,
    ):
        """
        Initialize the AgentStep with a language model and a callback function.

        Args:
            llm (BaseChatOpenAI): The language model to use.
            step_callback (Callable): A callback function to handle step results.
        """
        self.llm = llm
        self.step_callback = step_callback
        self.summarized_files_db = summarized_files_db
        self.graph: CompiledGraph | None = None

    def _print_step(
        self,
        step_name: str | None = None,
        step_output: str | None = None,
        token_usage: dict | object | None = None,
    ):
        if self.step_callback is None:
            return
        # convert token_usage to dict
        if token_usage is not None and not isinstance(token_usage, dict):
            token_usage = vars(token_usage)
            token_usage = {**DEFAULT_TOKEN_USAGE, **token_usage}
        step_callback = self.step_callback
        step_callback(
            step_name=step_name,
            step_output=step_output,
            token_usage=token_usage,
        )

    def compile(self, repo_path: str, gitignore_path: str, **kwargs):
        """
        Compile the agent step with the given repository and gitignore paths.

        Args:
            repo_path (str): The path to the repository.
            gitignore_path (str): The path to the .gitignore file.
            **kwargs: derived class may pass more arguments to implmented _compile(), that is,
                what **kwargs is depends on derived class
        """
        self._compile(repo_path, gitignore_path, **kwargs)
    
    @abstractmethod
    def _compile(self, repo_path: str, gitignore_path: str, **kwargs):
        """
        Abstract method to compile the agent step.

        Args:
            repo_path (str): The path to the repository.
            gitignore_path (str): The path to the .gitignore file.
        """
        pass

    def _go_graph(self, input: dict) -> dict:
        input = {
            **input, 
            "llm": self.llm,
            "step_output_callback": self.step_callback,
        }
        for s in self.graph.stream(
            input=input, 
            stream_mode="values",
            config={"recursion_limit": 500},
        ):
            print(s)

        return s



