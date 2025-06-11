
import os
from .common_agent_2step import CommonAgentTwoSteps

EVALUATION_README_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. Your task is to analyze the README file found in the repository and provide a detailed evaluation based on the following criteria:
1. **Clarity**: Is the purpose of the project clearly stated?
2. **Installation Instructions**: Are there clear instructions on how to install the project?
3. **Usage Instructions**: Are there examples or instructions on how to use the project?
4. **Contributing Guidelines**: Are there guidelines for contributing to the project?
5. **License Information**: Is the license clearly stated?
6. **Overall Quality**: Provide an overall assessment of the README file's quality.
"""

class EvaluationREADMETask:
    def __init__(self, llm, repo_path: str, gitignore_path: str):
        self.llm = llm
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path
        
    def _find_readme_file(self) -> str | None:
        """
        Search for a README file in the repository directory.
        """
        readme_files = [
            "readme.md",
            "readme.rst",
            "readme.txt",
            "readme",
        ]
        for _, _, files in os.walk(self.repo_path):
            for file in files:
                if file.lower() in readme_files:
                    return file
                
        return None
    
    def _collect_installation_instructions(self) -> str:
        return ""
    
    def _collect_license_information(self) -> str:
        return ""

    def evaluate(self):
        readme_file = self._find_readme_file()
        if readme_file is None:
            raise ValueError("No README file found in the repository.")
        

