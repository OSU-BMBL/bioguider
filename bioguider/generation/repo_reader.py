from __future__ import annotations

import os
from typing import Dict, Optional, List, Tuple


class RepoReader:
    def __init__(self, repo_path: str, gitignore_path: Optional[str] = None):
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path

    def read_files(self, rel_paths: List[str]) -> Tuple[Dict[str, str], List[str]]:
        contents: Dict[str, str] = {}
        missing: List[str] = []
        for rel in rel_paths:
            abs_path = os.path.join(self.repo_path, rel)
            if not os.path.isfile(abs_path):
                missing.append(rel)
                continue
            try:
                with open(abs_path, "r", encoding="utf-8") as fobj:
                    contents[rel] = fobj.read()
            except Exception:
                missing.append(rel)
        return contents, missing

    def read_default_targets(self) -> Tuple[Dict[str, str], List[str]]:
        # Common targets we may need to modify
        candidates = [
            "README.md",
            "README.rst",
            "vignettes/install.Rmd",
            "vignettes/install_v5.Rmd",
        ]
        return self.read_files(candidates)


