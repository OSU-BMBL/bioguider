import os
from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
from bioguider.agents.agent_utils import read_directory, read_file, summarize_file

class agent_tool:
    def __init__(
        self,
        llm: BaseChatOpenAI | None = None,
        token_usage_callback:Callable[[dict], None] = None,
    ):
        self.llm = llm
        self.token_usage_callback = token_usage_callback

    def _print_token_usage(self, token_usage: dict):
        if self.token_usage_callback is not None:
            self.token_usage_callback(token_usage=token_usage)

class read_file_tool:
    """ read file
Args:
    file_path str: file path
Returns:
    A string of file content, if the file does not exist, return None. 
        """
    def __init__(self, repo_path: str | None = None):
        self.repo_path = repo_path if repo_path is not None else ""
    
    def run(self, file_path: str) -> str | None:
        if file_path is None:
            return None
        file_path = file_path.strip()
        if self.repo_path is not None and self.repo_path not in file_path:
            file_path = os.path.join(self.repo_path, file_path)
        if not os.path.isfile(file_path):
            return None
        return read_file(file_path)

class summarize_file_tool(agent_tool):
    """ read and summarize the file
Args:
    file_path str: file path
Returns:
    A string of summarized file content, if the file does not exist, return None.         
        """
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        repo_path: str | None = None,
        token_usage_callback: Callable | None = None,
        detailed_level: int | None = 6,
    ):
        super().__init__(llm=llm, token_usage_callback=token_usage_callback)
        self.repo_path = repo_path
        detailed_level = detailed_level if detailed_level is not None else 6
        detailed_level = detailed_level if detailed_level > 0 else 1
        detailed_level = detailed_level if detailed_level <= 10 else 10
        self.detailed_level = detailed_level

    def run(self, file_path: str) -> str | None:
        if file_path is None:
            return None
        file_path = file_path.strip()
        abs_file_path = file_path
        if self.repo_path is not None and self.repo_path not in abs_file_path:
            abs_file_path = os.path.join(self.repo_path, abs_file_path)
        if not os.path.isfile(abs_file_path):
            return f"{file_path} is not a file."
        file_content = read_file(abs_file_path)
        summarized_content, token_usage = summarize_file(
            self.llm, abs_file_path, file_content, self.detailed_level
        )
        if self.token_usage_callback is not None:
            self.token_usage_callback(token_usage)
        return f"summarized content of file {file_path}: " + summarized_content
    
class read_directory_tool:
    """Reads the contents of a directory, including files and subdirectories in it..
Args:
    dir_path (str): Path to the directory.
Returns:
    a string containing file and subdirectory paths found within the specified depth.
    """
    def __init__(
        self, 
        repo_path: str | None = None,
        gitignore_path: str | None = None,
    ):
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path if gitignore_path is not None else ""

    def run(self, dir_path):
        dir_path = dir_path.strip()
        full_path = dir_path
        if full_path == "." or full_path == "..":
            return f"Please skip this folder {dir_path}"
        if self.repo_path not in full_path:
            full_path = os.path.join(self.repo_path, full_path)
        files = read_directory(full_path, gitignore_path=self.gitignore_path, level=1)
        if files is None:
            return "N/A"
        file_pairs = [(f, "f" if os.path.isfile(os.path.join(full_path, f)) else "d") for f in files]
        dir_structure = ""
        for f, f_type in file_pairs:
            dir_structure += f"{os.path.join(dir_path, f)} - {f_type}\n"
        return f"The 1-level content of directory {dir_path}: " + dir_structure
