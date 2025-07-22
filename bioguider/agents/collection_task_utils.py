import os
from typing import Callable, Optional, TypedDict
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
import logging

from bioguider.agents.agent_tools import agent_tool
from bioguider.agents.agent_utils import read_file, summarize_file
from bioguider.agents.peo_common_step import PEOWorkflowState
from bioguider.agents.common_agent import CommonAgent
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.database.summarized_file_db import SummarizedFilesDb
from bioguider.utils.constants import MAX_FILE_LENGTH

logger = logging.getLogger(__name__)

class CollectionWorkflowState(TypedDict):
    llm: Optional[BaseChatOpenAI]
    step_output_callback: Optional[Callable]
    
    intermediate_steps: Optional[str]
    step_output: Optional[str]
    step_analysis: Optional[str]
    step_thoughts: Optional[str]
    plan_actions: Optional[list[dict]]

    goal_item: Optional[str]
    final_answer: Optional[str]

RELATED_FILE_GOAL_ITEM = """
Your task is to determine whether the file is related to **{goal_item}**.

{related_file_description} 
"""

CHECK_FILE_RELATED_USER_PROMPT = ChatPromptTemplate.from_template("""
You are given a summary of a file’s content.  

{goal_item_desc}

Here is the file summary:  
```
{summarized_file_content}
```

### **Question:**  
Does this file appear to contain related information?

---

### **Output Format:** 
Respond with exactly two parts:
1. A single word: Yes or No (indicating if the file meets the goal criteria)
2. One brief explanatory sentence.
For example: Yes. This file is a compiled binary file, so, it is related to the compiled standalone file (goal item).
""")

class CheckFileRelatedResult(BaseModel):
    is_related: str = Field(description="A string conclusion specify if the provided file is related. The string value contains two parts:\n 1. A single word: Yes or No (indicating if the file meets the goal criteria).\n 2. One brief explanatory sentence.")

class check_file_related_tool(agent_tool):
    """ Check if the file is related to the goal item
Args:
    file_path str: file path
Returns:
    str: A string conclusion. The string conclusion contains two parts:\n 1. A single word: Yes or No (indicating if the file meets the goal criteria).\n 2. One brief explanatory sentence.
    """ 
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        repo_path: str,
        goal_item_desc: str,
        output_callback: Callable | None = None,
        summarize_instruction: str | None = None,
        summarize_level: int | None = 6,
        summarized_files_db: SummarizedFilesDb | None = None,
    ):
        super().__init__(llm=llm, output_callback=output_callback)
        self.repo_path = repo_path
        self.goal_item_desc = goal_item_desc
        self.summarize_instruction = summarize_instruction \
            if summarize_instruction is not None else "N/A"
        self.summarize_level = summarize_level
        self.summarized_files_db = summarized_files_db

    def run(self, file_path: str) -> str:
        if not self.repo_path in file_path:
            file_path = os.path.join(self.repo_path, file_path)
        if not os.path.isfile(file_path):
            return "Can't read file"
        
        check_prompts = None
        try:
            file_content = read_file(file_path)
        except UnicodeDecodeError as e:
            logger.error(str(e))
            check_prompts = "Can't summarize binary file, please decide according to file name and extension."
        except Exception as e:
            logger.error(str(e))
            check_prompts = "Failed to summarize file, please decide according to file name and extension."
        if check_prompts is None and file_content is None:
            return "Failed to read file"
        if check_prompts is not None:
            summarized_content = check_prompts
        else:
            if len(file_content) > MAX_FILE_LENGTH:
                file_content = file_content[:MAX_FILE_LENGTH]
            summarized_content, token_usage = summarize_file(
                llm=self.llm, 
                name=file_path, 
                content=file_content, 
                level=self.summarize_level,
                summary_instructions=self.summarize_instruction,
                db=self.summarized_files_db,
            )
            if summarized_content is None:
                return "Failed to summarize file"
            self._print_token_usage(token_usage)
        
        prompt = CHECK_FILE_RELATED_USER_PROMPT.format(
            goal_item_desc=self.goal_item_desc,
            summarized_file_content=summarized_content,
        )

        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning = agent.go(
            system_prompt=prompt,
            instruction_prompt="Now, please check if the file is related to the goal item.",
            schema=CheckFileRelatedResult,
        )
        # res: AIMessage = self.llm.invoke([("human", prompt)])
        res: CheckFileRelatedResult = res
        out = res.is_related
        
        self._print_step_output(step_output=reasoning)
        self._print_token_usage(token_usage)
        return res.is_related
        