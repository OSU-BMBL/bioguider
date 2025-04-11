
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from bioguider.agents.common_agent import (
    CommonAgentResult,
    RetryException,
)


IDENTIFICATION_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert in both biomedical research and software development. Given the file structure of a code repository, your task is to determine the primary programming language and the project type.
The input is the project’s directory structure (f - means file, d - means directory):
{project_structure}

Please analyze the structure and return your answer in exactly the following format:
1. language: <one of: python, R, unsupported language>
2. type: <one of: package, application, pipeline, unknown type>
Definitions:
package: A reusable Python or R library that is designed to be imported by other software.
application: A standalone Python or R program that users can execute directly.
pipeline: A biomedical data processing workflow that orchestrates multiple tools or steps to transform or analyze data.
unknown type: Use this when the project purpose cannot be reliably inferred from the file structure.

Instructions:
1. Only use the file and directory names to infer the language and type.
2. Be conservative in your judgment—if unsure, choose unsupported language or unknown type.
""")

class EvaluationIdentificationAgentResult(CommonAgentResult):
    language: str = Field(description="a string representing the project main language: **python**, **R** or **unsupported language**")
    type: str = Field(description="a string representing the project type: ")

def get_system_prompt(project_structure: str):
    return IDENTIFICATION_SYSTEM_PROMPT.format(
        project_structure=project_structure,
    )

def postprocess_validate_result(res: EvaluationIdentificationAgentResult):
    if res.language not in ["python", "R", "unsupported language"]:
        return RetryException(f"You provided wrong language {res.language}. The language is expected to be **python**, **R** or **unsupported language**")
    if res.type not in ["application", "package", "pipeline", "unknown type"]:
        return RetryException(f"You provided wrong project type {res.type}. The type is expected to be **application**, **package**, **pipeline** or **unknown type**")
    
    return res