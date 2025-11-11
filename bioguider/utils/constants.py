
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

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
MAX_STEP_COUNT=3*10

class ProjectLevelEvaluationREADMEResult(BaseModel):
    project_level: Optional[bool]=Field(description="A boolean value specifying if the README file is **project-level** README. TRUE: project-level, FALSE, folder-level")

class StructuredEvaluationREADMEResult(BaseModel):
    available_score: Optional[bool]=Field(description="A boolean value, Is the README accessible and present?")
    readability_score: Optional[int]=Field(description="A number between 0 and 100 representing the readability quality rating.")
    readability_suggestions: Optional[str]=Field(description="Suggestions to improve readability if necessary")
    project_purpose_score: Optional[bool]=Field(description="A boolean value. Is the project's goal or function clearly stated?")
    project_purpose_suggestions: Optional[str]=Field(description="Suggestions if not clear")
    hardware_and_software_spec_score: Optional[int]=Field(description="A number between 0 and 100 representing the hardware and software spec and compatibility description quality rating.")
    hardware_and_software_spec_suggestions: Optional[str]=Field(description="Suggestions if not clear")
    dependency_score: Optional[int]=Field(description="A number between 0 and 100 representing the dependencies quality rating.")
    dependency_suggestions: Optional[str]=Field(description="Suggestions if dependencies are not clearly stated")
    license_score: Optional[bool]=Field(description="A boolean value, Are contributor or maintainer details provided?")
    license_suggestions: Optional[str]=Field(description="Suggestions to improve license information")
    contributor_author_score: Optional[bool]=Field(description="A boolean value. are contributors or author included?")
    overall_score: int=Field(description="A number between 0 and 100 representing the overall quality rating.")

class FreeProjectLevelEvaluationREADMEResult(BaseModel):
    available: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestion for the availability of the README file")
    readability: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestion for the readability of the README file")
    project_purpose: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestion for the project purpose of the README file")
    hardware_and_software_spec: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestion for the hardware and software spec and compatibility description of the README file")
    dependency: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestion for the dependencies of the README file")
    license: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestion for the license information of the README file")
    contributor_author: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestion for the contributor and author information of the README file")
    overall_score: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestion for the overall score of the README file")

class FreeFolderLevelEvaluationREADMEResult(BaseModel):
    score: Optional[str]=Field(description="An overall score")
    key_strengths: Optional[str]=Field(description="A string specifying the key strengths of README file.")
    overall_improvement_suggestions: Optional[list[str]]=Field(description="A list of overall improvement suggestions")

class EvaluationREADMEResult(BaseModel):
    project_level: bool
    structured_evaluation: StructuredEvaluationREADMEResult | None
    free_evaluation: FreeProjectLevelEvaluationREADMEResult | FreeFolderLevelEvaluationREADMEResult | None
    structured_reasoning_process: str | None
    free_reasoning_process: str | None


class StructuredEvaluationInstallationResult(BaseModel):
    install_available: Optional[bool]=Field(description="A boolean value. Is the installation documents accessible and present?")
    install_tutorial: Optional[bool]=Field(description="A boolean value. Is the installation tutorial provided?")
    dependency_number: Optional[int]=Field(description="A number. It is the number of dependencies that are required to install.")
    dependency_suggestions: Optional[str]=Field(description="A string value. It is the specific improvements if necessary, such as missing dependencies")
    compatible_os: Optional[bool]=Field(description="A boolean value. Is compatible operating system described?")
    overall_score: Optional[int]=Field(description="A number between 0 and 100 representing the overall quality rating.")
    hardware_requirements: Optional[bool]=Field(description="A boolean value. Is the hardware requirements described?")

class FreeEvaluationInstallationResult(BaseModel):
    ease_of_access: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestions for the ease of access of the installation information")
    clarity_of_dependency: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestions for the clarity of dependency specification")
    hardware_requirements: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestions for the hardware requirements")
    installation_guide: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestions for the installation guide")
    compatible_os: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestions for the compatible operating system")
    overall_score: Optional[list[str]]=Field(description="markdown texts including newlines that contains detailed assessment and detailed suggestions for the overall score of the installation")

class EvaluationInstallationResult(BaseModel):
    structured_evaluation: StructuredEvaluationInstallationResult | None
    free_evaluation: FreeEvaluationInstallationResult | None
    structured_reasoning_process: str | None
    free_reasoning_process: str | None

class SoftwarePackageContentResult(BaseModel):
    compiled_standalone_software: Optional[bool] = Field(description="A boolean value. Does it provide the compiled standalone software?")
    source_code: Optional[bool] = Field(description="A boolean value. Does it provide the source code?")
    demo_dataset: Optional[bool] = Field(description="A boolean value. Does it provide the demo dataset?")

class DemoInstructionsResult(BaseModel):
    run_on_data_instruction: Optional[bool] = Field(description="A boolean value. Does it provide instructions on how to run on provided data?")
    run_on_custom_instruction: Optional[bool] = Field(description="A boolean value. Does it provide instructions on how to run on custom data?")
    expected_output_description: Optional[bool] = Field(description="A boolean value. Does it provide the description of expected output?")

class EvaluationSubmissionRequirementsResult(BaseModel):
    compiled_standalone_software: bool | None
    source_code: bool | None
    demo_dataset: bool | None
    run_on_data_instruction: bool | None
    run_on_custom_instruction: bool | None
    expected_output_description: bool | None
    complete_readme: bool | None
    software_dependency: bool | None
    install_tutorial: bool | None
    license: bool | None
    hardware_requirements: bool | None
    compatible_os: bool | None
