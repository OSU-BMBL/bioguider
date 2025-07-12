import os
from pathlib import Path

from bioguider.agents.prompt_utils import CollectionGoalItemEnum
from bioguider.utils.constants import ProjectMetadata
from bioguider.utils.gitignore_checker import GitignoreChecker

from ..agents.identification_task import IdentificationTask
from ..rag.rag import RAG
from ..utils.file_utils import parse_repo_url
from ..database.summarized_file_db import SummarizedFilesDb
from ..agents.evaluation_task import EvaluationREADMETask
from ..agents.collection_task import CollectionTask

class EvaluationManager:
    def __init__(self, llm, step_callback):
        self.rag = None
        self.llm = llm
        self.step_callback = step_callback
        self.repo_url: str | None = None
        self.project_metadata: ProjectMetadata | None = None

    def prepare_repo(self, repo_url: str):
        self.repo_url = repo_url
        self.rag = RAG()
        self.rag.initialize_db_manager()
        self.rag.prepare_retriever(repo_url_or_path=repo_url)

        author, repo_name = parse_repo_url(repo_url)
        self.summary_file_db = SummarizedFilesDb(author, repo_name)

    def identify_project(self) -> ProjectMetadata:
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

        self.project_metadata = ProjectMetadata(
            url=self.repo_url,
            project_type=project_type,
            primary_language=language,
            repo_name=meta_data["name"] if "name" in meta_data else "",
            description=meta_data["description"] if "description" in meta_data else "",
            owner=meta_data["owner"] if "owner" in meta_data else "",
            license=meta_data["license"] if "license" in meta_data else "",
        )
        return self.project_metadata
    
    def evaluate_readme(self):
        task = EvaluationREADMETask(
            llm=self.llm,
            repo_path=self.rag.repo_dir,
            gitignore_path=Path(self.rag.repo_dir, ".gitignore"),
            meta_data=self.project_metadata,
            step_callback=self.step_callback,
        )
        readme_files = self._find_readme_files()
        results = task.evaluate(readme_files)
        return results
    
    def evaluate_tutorial(self):
        task = CollectionTask(
            llm=self.llm,
            step_callback=self.step_callback,
        )
        task.compile(
            repo_path=self.rag.repo_dir,
            gitignore_path=Path(self.rag.repo_dir, ".gitignore"),
            db=self.summary_file_db,
            goal_item=CollectionGoalItemEnum.Tutorial.name,
        )
        s = task.collect()
        if s is None or 'final_answer' not in s:
            return None
        
    def _find_readme_files(self) -> list[str]:
        """
        Search for a README file in the repository directory.
        """
        possible_readme_files = [
            "readme.md",
            "readme.rst",
            "readme.txt",
            "readme",
        ]
        repo_path = self.rag.repo_dir
        gitignore_path = Path(repo_path, ".gitignore")
        gitignore_checker = GitignoreChecker(
            directory=self.repo_path, gitignore_path=gitignore_path
        )
        found_readme_files = gitignore_checker.check_files_and_folders(
            check_file_cb=lambda root_dir, relative_path: Path(relative_path).name.lower() in possible_readme_files,
        )
                
        return found_readme_files
    

