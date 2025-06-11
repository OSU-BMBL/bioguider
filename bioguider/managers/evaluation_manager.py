
from pathlib import Path

from ..agents.identification_task import IdentificationTask
from ..rag.rag import RAG
from ..utils.file_utils import parse_repo_url
from ..database.summarized_file_db import SummarizedFilesDb

class EvaluationManager:
    def __init__(self, llm, step_callback):
        self.rag = None
        self.llm = llm
        self.step_callback = step_callback

    def prepare_repo(self, repo_url: str):
        self.rag = RAG()
        self.rag.initialize_db_manager()
        self.rag.prepare_retriever(repo_url_or_path=repo_url)

        author, repo_name = parse_repo_url(repo_url)
        self.summary_file_db = SummarizedFilesDb(author, repo_name)

    def identify_project(self):
        repo_path = self.rag.repo_dir
        gitignore_path = Path(repo_path, ".gitignore")

        identfication_task = IdentificationTask(
            llm=self.llm,
            step_callback=self.step_callback,
        )
        identfication_task.compile(
            repo_path=repo_path,
            gitignore_path=gitignore_path,
            db=self.summary_file_db,
        )
        language = identfication_task.identify_primary_language()
        project_type = identfication_task.identify_project_type()
        meta_data = identfication_task.identify_meta_data()

        return project_type, language, meta_data


