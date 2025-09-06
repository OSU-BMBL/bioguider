from pathlib import Path
import logging

from .gitignore_checker import GitignoreChecker
from .file_handler import FileHandler
from ..database.code_structure_db import CodeStructureDb

logger = logging.getLogger(__name__)

class CodeStructureBuilder:
    def __init__(
        self, 
        repo_path: str, 
        gitignore_path: str,
        code_structure_db: CodeStructureDb,
    ):
        self.repo_path = repo_path
        self.gitignore_checker = GitignoreChecker(repo_path, gitignore_path)
        self.file_handler = FileHandler(repo_path)
        self.code_structure_db = code_structure_db

    def build_code_structure(self):
        files = self.gitignore_checker.check_files_and_folders()
        for file in files:
            if not file.endswith(".py"):
                continue
            logger.info(f"Building code structure for {file}")
            file_handler = FileHandler(Path(self.repo_path) / file)
            functions_and_classes = file_handler.get_functions_and_classes()
            # fixme: currently, we don't extract reference graph for each function or class
            for function_or_class in functions_and_classes:
                self.code_structure_db.insert_code_structure(
                    function_or_class[0], # name
                    file,
                    function_or_class[2], # start line number
                    function_or_class[3], # end line number
                    function_or_class[1], # parent name
                    function_or_class[4], # doc string
                    function_or_class[5], # params
                )
        

