
from pathlib import Path
import pytest

from bioguider.utils.gitignore_checker import GitignoreChecker

def test_GitignoreChecker():
    gitignore_checker = GitignoreChecker(
        directory="./data/repos/BMBL-analysis-notebooks",
        gitignore_path="./data/repos/BMBL-analysis-notebooks/.gitignore",
    )
    readme_files = [
        "readme.md",
        "readme.rst",
        "readme.txt",
        "readme",
    ]
    def is_file_readme(root_dir: str, relative_path: str):
        fn = Path(relative_path).name.lower()
        return fn in readme_files
    files = gitignore_checker.check_files_and_folders(
        check_file_cb=lambda root_dir, relative_path: Path(relative_path).name.lower() in readme_files,
    )

    assert len(files) > 1

