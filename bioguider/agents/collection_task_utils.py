import os
from typing import Callable, Optional, TypedDict
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from bioguider.agents.agent_tools import agent_tool
from bioguider.agents.agent_utils import read_file, summarize_file
from bioguider.agents.peo_common_step import PEOWorkflowState
from bioguider.agents.common_agent import CommonAgent
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps


class CollectionWorkflowState(TypedDict):
    llm: Optional[BaseChatOpenAI]
    step_output_callback: Optional[Callable]
    
    intermediate_steps: Optional[str]
    step_output: Optional[str]
    step_analysis: Optional[str]
    step_thoughts: Optional[str]
    plan_actions: Optional[list[dict]]

    goal_item: Optional[str]

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
Respond with a single word: "Yes" or "No" to indicate whether the file is related to the goal item.
Do not include any additional text, explanation, or formatting.
""")

class CheckFileRelatedResult(BaseModel):
    is_related: bool = Field(description="True if the file is related to the goal item, False otherwise.")

class check_file_related_tool(agent_tool):
    """ Check if the file is related to the goal item
Args:
    file_path str: file path
Returns:
    bool: True if the file is related to the goal item, False otherwise.
    """ 
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        repo_path: str,
        goal_item_desc: str,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm=llm, output_callback=output_callback)
        self.repo_path = repo_path
        self.goal_item_desc = goal_item_desc

    def run(self, file_path: str) -> str:
        if not self.repo_path in file_path:
            file_path = os.path.join(self.repo_path, file_path)
        if not os.path.isfile(file_path):
            return "Can't read file"
        file_content = read_file(file_path)
        if file_content is None:
            return "Failed to read file"
        summarized_content, token_usage = summarize_file(self.llm, file_path, file_content, 6)
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
        if out:
            return "Yes, the file is related to the goal item."
        else:
            return "No, the file **is not** related to the goal item."
        