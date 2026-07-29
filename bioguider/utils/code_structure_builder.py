from pathlib import Path
import logging

from bioguider.utils.r_file_handler import RFileHandler

from .gitignore_checker import GitignoreChecker
from .python_file_handler import PythonFileHandler
from ..database.code_structure_db import CodeStructureDb
from ..rag.config import configs

logger = logging.getLogger(__name__)

class CodeStructureBuilder:
    def __init__(
        self, 
        repo_path: str | Path, 
        gitignore_path: str | Path,
        code_structure_db: CodeStructureDb,
    ):
        self.repo_path = str(repo_path)
        self.gitignore_checker = GitignoreChecker(
            directory=repo_path,
            gitignore_path=str(gitignore_path),
            exclude_dir_patterns=configs["file_filters"]["excluded_dirs"],
            exclude_file_patterns=configs["file_filters"]["excluded_files"],
        )
        self.file_handler = PythonFileHandler(repo_path)
        self.code_structure_db = code_structure_db

    def build_code_structure(self):
        if self.code_structure_db.is_database_built():
            return
        files = self.gitignore_checker.check_files_and_folders()
        for file in files:
            if not file.endswith(".py") and not file.endswith(".R"):
                continue
            logger.info(f"Building code structure for {file}")
            if file.endswith(".py"):
                file_handler = PythonFileHandler(Path(self.repo_path) / file)
            else:
                file_handler = RFileHandler(Path(self.repo_path) / file)
            try:
                functions_and_classes = file_handler.get_functions_and_classes()
            except Exception as e:
                logger.error(f"Error getting functions and classes for {file}: {e}")
                continue
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

            # argparse command-line interface (Python files only, for now)
            if file.endswith(".py"):
                try:
                    cli_iface = file_handler.get_cli_interface()
                except Exception as e:
                    logger.error(f"Error getting CLI interface for {file}: {e}")
                    cli_iface = None
                if cli_iface is not None:
                    for arg in cli_iface.arguments:
                        self.code_structure_db.insert_cli_argument(
                            path=file,
                            dest=arg.dest,
                            option_strings=arg.option_strings,
                            prog=cli_iface.prog,
                            description=cli_iface.description,
                            subcommand=arg.subcommand,
                            arg_type=arg.arg_type,
                            default_value=arg.default,
                            action=arg.action,
                            nargs=arg.nargs,
                            choices=arg.choices,
                            required=arg.required,
                            metavar=arg.metavar,
                            help=arg.help,
                            lineno=arg.lineno,
                        )
        

