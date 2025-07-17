import os
from pathlib import Path
import logging
from typing import Callable, Optional
from abc import ABC, abstractmethod
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from bioguider.agents.agent_utils import read_file
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE, ProjectMetadata
from .common_agent_2step import CommonAgentTwoSteps, CommonAgentTwoChainSteps
from .common_agent import CommonConversation
from ..utils.pyphen_utils import PyphenReadability
from ..utils.gitignore_checker import GitignoreChecker
from .evaluation_task import EvaluationTask
from .agent_utils import read_file


logger = logging.getLogger(__name__)

EVALUATION_INSTALLATION_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of **installation instructions** in software repositories.
Your task is to analyze the provided content of installation-related files and generate a **comprehensive, structured quality report**.

---

### Evaluation Criteria

Please assess the installation information using the following criteria. For each, provide a concise evaluation and specific feedback:

1. **Ease of Access**
   * Is the installation information clearly presented and easy to locate within the repository?
   * Is it included in a top-level README, a dedicated INSTALL.md file, or other accessible locations?

2. **Clarity of Dependency Specification**
   * Are all software and library dependencies clearly listed?
   * Are installation methods (e.g., `pip`, `conda`, `apt`) for those dependencies explicitly provided?

3. **Hardware Requirements**
   * Does the documentation specify hardware needs (e.g., GPU, memory, OS) if relevant?

4. **Step-by-Step Installation Guide**
   * Is there a clear, ordered set of instructions for installing the software?
   * Are example commands or configuration steps provided to help users follow along?

---

### Output Format

Your response **must exactly follow** the structure below:

```
**FinalAnswer**
**Overall Score:** [Poor / Fair / Good / Excellent]  
**Ease of Access:** <your comments>  
**Clarity of Dependency Specification:** <your comments>  
**Hardware Requirements:** <your comments>  
**Installation Guide:** <your comments>  
```

---

### Installation Files Provided:
{installation_file_contents}

"""

class EvaluationInstallationResult(BaseModel):
    ease_of_access: Optional[str]=Field(description="Is the installation information easy to access")
    score: Optional[str]=Field(description="An overall score, could be Poor, Fair, Good or Excellent")
    clarity_of_dependency: Optional[str]=Field(description="Are all dependencies clearly listed")
    hardware_requirements: Optional[str]=Field(description="Are all hardware requirements clearly specified")
    installation_guide: Optional[str]=Field(description="Is there a clear, ordered set of instructions for installing the software")

EvaluationInstallationResultSchema = {
    "title": "EvaluationREADMEResult",
    "type": "object",
    "properties": {
        "ease_of_access": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Is the installation information easy to access",
            "title": "Ease of Access"
        },
        "score": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "An overall score, could be Poor, Fair, Good or Excellent",
            "title": "Score"
        },
        "clarity_of_dependency": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Are all dependencies clearly listed",
            "title": "Clarity of Dependency",
        },
        "hardware_requirements": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Are all hardware requirements clearly specified",
            "title": "Hardware Requirements"
        },
        "installation_guide": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Is there a clear, ordered set of instructions for installing the software",
            "title": "Installation Guide"
        }
    },
    "required": ["ease_of_access", "score", "clarity_of_dependency", "hardware_requirements", "installation_guide"]
}

class EvaluationInstallationTask(EvaluationTask):
    def __init__(
        self, 
        llm, 
        repo_path,
        gitignore_path, 
        meta_data = None, 
        step_callback = None,
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback)

    def _evaluate(self, files: list[str] | None = None):
        if files is None or len(files) == 0:
            return None
        
        files_content = ""
        for f in files:
            content = read_file(os.path.join(self.repo_path, f))
            files_content += f"""
{f} content:
{content}

"""
        system_prompt = ChatPromptTemplate.from_template(EVALUATION_INSTALLATION_SYSTEM_PROMPT).format(
            installation_file_contents=files_content
        )
        agent = CommonAgentTwoChainSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Before arriving at the conclusion, clearly explain your reasoning step by step. Now, let's begin the evaluation.",
            schema=EvaluationInstallationResultSchema,
        )
        res = EvaluationInstallationResult(**res)
        evaluation = {
            "score": res.score,
            "ease_of_access": res.ease_of_access,
            "hardware_requirements": res.hardware_requirements,
            "clarity_of_dependency": res.clarity_of_dependency,
            "installation_guide": res.installation_guide,
            "reasoning_process": reasoning_process,
        }
        return evaluation, token_usage
        
