
from enum import Enum
from typing import Optional

DEFAULT_TOKEN_USAGE = {
    "total_tokens": 0,
    "completion_tokens": 0,
    "prompt_tokens": 0,
}

class ProjectTypeEnum(Enum):
    application="application"
    package="package"
    pipeline="pipeline"
    unknown="unknown type"

class PrimaryLanguageEnum(Enum):
    python="python"
    R="R"
    unknown="unknown type"

class ProjectMetadata:
    def __init__(
        self,
        url: str,
        project_type: ProjectTypeEnum,
        primary_language: PrimaryLanguageEnum,
        repo_name: str=None,
        owner: Optional[str]=None,
        description: Optional[str]=None,
        license: Optional[str]=None,
    ):
        self.url = url
        self.project_type = project_type
        self.primary_language = primary_language
        self.repo_name = repo_name
        self.owner = owner
        self.description = description
        self.license = license

MAX_FILE_LENGTH=10 *1024 # 10K
MAX_SENTENCE_NUM=20