import os
from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
from bioguider.database.summarized_file_db import SummarizedFilesDb
from bioguider.utils.file_utils import get_file_type
from bioguider.agents.agent_utils import read_directory, read_file, summarize_file

class agent_tool:
    def __init__(
        self,
        llm: BaseChatOpenAI | None = None,
        output_callback:Callable[[dict], None] = None,
    ):
        self.llm = llm
        self.output_callback = output_callback

    def _print_token_usage(self, token_usage: dict):
        if self.output_callback is not None:
            self.output_callback(token_usage=token_usage)
    def _print_step_output(self, step_output: str):
        if self.output_callback is not None:
            self.output_callback(step_output=step_output)

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
        output_callback: Callable | None = None,
        detailed_level: int | None = 6,
        db: SummarizedFilesDb | None = None,
        summaize_instruction: str = "",
    ):
        super().__init__(llm=llm, output_callback=output_callback)
        self.repo_path = repo_path
        detailed_level = detailed_level if detailed_level is not None else 6
        detailed_level = detailed_level if detailed_level > 0 else 1
        detailed_level = detailed_level if detailed_level <= 10 else 10
        self.detailed_level = detailed_level
        self.summary_file_db = db
        self.summarize_instruction = summaize_instruction

    def _retrive_from_summary_file_db(self, file_path: str) -> str | None:
        if self.summary_file_db is None:
            return None
        return self.summary_file_db.select_summarized_text(
            file_path=file_path,
            instruction=self.summarize_instruction,
            summarize_level=self.detailed_level,
        )
    def _save_to_summary_file_db(self, file_path: str, summarized_text: str, token_usage: dict):
        if self.summary_file_db is None:
            return
        self.summary_file_db.upsert_summarized_file(
            file_path=file_path,
            instruction=self.summarize_instruction,
            summarize_level=self.detailed_level,
            summarized_text=summarized_text,
            token_usage=token_usage,
        )
    def run(self, file_path: str) -> str | None:
        if file_path is None:
            return None
            
        file_path = file_path.strip()
        abs_file_path = file_path
        if self.repo_path is not None and self.repo_path not in abs_file_path:
            abs_file_path = os.path.join(self.repo_path, abs_file_path)
        if not os.path.isfile(abs_file_path):
            return f"{file_path} is not a file."
        summarized_content = self._retrive_from_summary_file_db(
            file_path=file_path
        )
        if summarized_content is not None:
            return f"summarized content of file {file_path}: " + summarized_content

        file_content = read_file(abs_file_path)
        file_content = file_content.replace("{", "{{").replace("}", "}}")
        summarized_content, token_usage = summarize_file(
            self.llm, abs_file_path, file_content, self.detailed_level,
            summary_instructions=self.summarize_instruction,
        )
        self._save_to_summary_file_db(
            file_path=file_path,
            summarized_text=summarized_content,
            token_usage=token_usage,
        )
        self._print_token_usage(token_usage)
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
        file_pairs = [(f, get_file_type(os.path.join(full_path, f)).value) for f in files]
        dir_structure = ""
        for f, f_type in file_pairs:
            dir_structure += f"{os.path.join(dir_path, f)} - {f_type}\n"
        return f"The 2-level content of directory {dir_path}: \n" + \
            f"{dir_structure if len(dir_structure) > 0 else 'No files and sub-directories in it'}"
