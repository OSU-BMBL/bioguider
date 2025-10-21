
import json
from json import JSONDecodeError
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union
from langchain_openai import AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
import logging

from pydantic import BaseModel, Field

from bioguider.utils.constants import DEFAULT_TOKEN_USAGE, MAX_FILE_LENGTH, MAX_SENTENCE_NUM
from bioguider.utils.file_utils import get_file_type
from bioguider.utils.utils import clean_action_input
from ..utils.gitignore_checker import GitignoreChecker
from ..database.summarized_file_db import SummarizedFilesDb
from bioguider.agents.common_conversation import CommonConversation
from bioguider.rag.config import configs

logger = logging.getLogger(__name__)

class PlanAgentResult(BaseModel):
    """ Identification Plan Result """
    actions: list[dict] = Field(description="a list of action dictionary, e.g. [{'name': 'read_file', 'input': 'README.md'}, ...]")

PlanAgentResultJsonSchema = {
    "title": "identification_plan_result",
    "description": "plan result",
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "description": """a list of action dictionary, e.g. [{'name': 'read_file', 'input': 'README.md'}, ...]""",
            "title": "Actions",
            "items": {"type": "object"}
        },
    },
    "required": ["actions"],
}

def get_openai():
    return get_llm(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=os.environ.get("OPENAI_MODEL"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=os.environ.get("OPENAI_API_VERSION"),
        azure_deployment=os.environ.get("OPENAI_DEPLOYMENT_NAME"),
        max_tokens=os.environ.get("OPENAI_MAX_OUTPUT_TOKEN"),
    )

def get_llm(
    api_key: str,
    model_name: str="gpt-4o",
    azure_endpoint: str=None,
    api_version: str=None,
    azure_deployment: str=None,
    temperature: float = 0.0,
    max_tokens: int = 16384,  # Set high by default - enough for any document type
):
    """
    Create an LLM instance with appropriate parameters based on model type and API version.
    
    Handles parameter compatibility across different models and API versions:
    - DeepSeek models: Use max_tokens parameter
    - GPT models (newer): Use max_completion_tokens parameter
    - GPT-5+: Don't support custom temperature (uses default)
    """
    
    if model_name.startswith("deepseek"):
        chat = ChatDeepSeek(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_name.startswith("gpt"):
        # Base parameters common to all GPT models
        llm_params = {
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
            "azure_deployment": azure_deployment,
            "model": model_name,
        }
        
        # Determine token limit parameter name based on API version
        # Newer APIs (2024-08+) use max_completion_tokens instead of max_tokens
        use_completion_tokens = api_version and api_version >= "2024-08-01-preview"
        token_param = "max_completion_tokens" if use_completion_tokens else "max_tokens"
        llm_params[token_param] = max_tokens
        
        # Handle temperature parameter based on model capabilities
        # GPT-5+ models don't support custom temperature values
        supports_temperature = not any(restricted in model_name for restricted in ["gpt-5", "o1", "o3"])
        if supports_temperature:
            llm_params["temperature"] = temperature
            
        chat = AzureChatOpenAI(**llm_params)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Validate the LLM instance with a simple test
    try:
        chat.invoke("Hi")
    except Exception as e:
        logger.error(f"Failed to initialize LLM {model_name}: {e}")
        return None
    
    return chat

def pretty_print(message, printout = True):
    if isinstance(message, tuple):
        title = message
    else:
        if isinstance(message.content, list):
            title = get_msg_title_repr(message.type.title().upper() + " Message", bold=is_interactive_env())
            if message.name is not None:
                title += f"\nName: {message.name}"

            for i in message.content:
                if i['type'] == 'text':
                    title += f"\n{i['text']}\n"
                elif i['type'] == 'tool_use':
                    title += f"\nTool: {i['name']}"
                    title += f"\nInput: {i['input']}"
            if printout:
                print(f"{title}")
        else:
            title = get_msg_title_repr(message.type.title() + " Message", bold=is_interactive_env())
            if message.name is not None:
                title += f"\nName: {message.name}"
            title += f"\n\n{message.content}"
            if printout:
                print(f"{title}")
    return title

HUGE_FILE_LENGTH = 10 * 1024 # 10K

def read_file(
    file_path: str | Path,
) -> str | None:
    file_path = str(file_path).strip()
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r') as f:
        content = f.read()
        return content

def write_file(file_path: str | Path, content: str):
    try:
        file_path = str(file_path).strip()
        with open(file_path, "w") as fobj:
            fobj.write(content)
            return True
    except Exception as e:
        logger.error(e)
        return False

def read_directory(
    dir_path: str | Path,
    gitignore_path: str,
    level: int=1,
) -> list[str] | None:
    dir_path = str(dir_path).strip()
    if not os.path.isdir(dir_path):
        return None
    gitignore_checker = GitignoreChecker(
        directory=dir_path,
        gitignore_path=gitignore_path,
        exclude_dir_patterns=configs["file_filters"]["excluded_dirs"],
        exclude_file_patterns=configs["file_filters"]["excluded_files"],
    )
    files = gitignore_checker.check_files_and_folders(level=level)
    return files


EVALUATION_SUMMARIZE_FILE_PROMPT = ChatPromptTemplate.from_template("""
You will be provided with the content of the file **{file_name}**:  

---

### **Summary Instructions**
{summary_instructions}
The content is lengthy. Please generate a concise summary ({sentence_num1}-{sentence_num2} sentences).

---

### **Important Instructions**
{summarize_prompt}

---

### **File Content**
Here is the file content:
{file_content}

---

Now, let's start to summarize.
""")


def summarize_file(
    llm: BaseChatOpenAI, 
    name: str | Path, 
    content: str | None = None, 
    level: int = 3,
    summary_instructions: str | None = None,
    summarize_prompt: str = "N/A",
    db: SummarizedFilesDb | None = None,
) -> Tuple[str, dict]:
    name = str(name).strip()
    if content is None:
        try:            
            with open(name, "r") as fobj:
                content = fobj.read()
        except Exception as e:
            logger.error(e)
            return ""
    # First, query from database
    if db is not None:
        res = db.select_summarized_text(name, summary_instructions, level)
        if res is not None:
            return res, {**DEFAULT_TOKEN_USAGE}

    file_content = content
    level = level if level > 0 else 1
    level = level if level < MAX_SENTENCE_NUM+1 else MAX_SENTENCE_NUM
    if len(file_content) > MAX_FILE_LENGTH:
        file_content = content[:MAX_FILE_LENGTH] + " ..."
    prompt = EVALUATION_SUMMARIZE_FILE_PROMPT.format(
        file_name=name, 
        file_content=file_content, 
        sentence_num1=level,
        sentence_num2=level+1,
        summary_instructions=summary_instructions \
            if summary_instructions is not None and len(summary_instructions) > 0 \
            else "N/A",
        summarize_prompt=summarize_prompt,
    )
    
    config = {"recursion_limit": 500}
    res: AIMessage = llm.invoke([("human", prompt)], config=config)
    out = res.content
    token_usage = {
        "prompt_tokens": res.usage_metadata["input_tokens"],
        "completion_tokens": res.usage_metadata["output_tokens"],
        "total_tokens": res.usage_metadata["total_tokens"],
    }
    if db is not None:
        db.upsert_summarized_file(
            file_path=name,
            instruction=summary_instructions,
            summarize_level=level,
            summarize_prompt=summarize_prompt,
            summarized_text=out,
            token_usage=token_usage,
        )
    
    return out, token_usage

  # Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]
    # Plan
    plan_actions: str

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\n"
        # Set plan_step
        kwargs["plan_actions"] = self.plan_actions
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        prompt = self.template.format(**kwargs)
        # print([prompt])
        return prompt
    
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            print(f"Warning: could not parse LLM output: `{llm_output}`, finishing chain...")
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        action_dict = None
        action_input_replaced = clean_action_input(action_input)
        try:
            action_dict = json.loads(action_input_replaced)
        except json.JSONDecodeError:
            pass
        if action_dict is None:
            # try using ast to parse input string
            import ast
            try:
                action_dict = ast.literal_eval(action_input_replaced)
                if not isinstance(action_dict, dict):
                    action_dict = None
            except Exception as e:
                logger.error(f"Error parsing action input: {action_input} -> {action_input_replaced}\n{e}")
                pass
        return AgentAction(
            tool=action, 
            tool_input=action_dict if action_dict is not None else action_input, 
            log=llm_output
        )

def get_tool_names_and_descriptions(tools: List[BaseTool]) -> str:
    tool_names = []
    tools_descriptions = ""
    for tool in tools:
        tools_descriptions += f"name: {tool.name}, description: {tool.description}\n"
        tool_names.append(tool.name)
    return str(tool_names), tools_descriptions

def generate_repo_structure_prompt(
    files: List[str],
    dir_path: str="",
) -> str:
    # Convert the repo structure to a string
    file_pairs = [(f, get_file_type(os.path.join(dir_path, f)).value) for f in files]
    repo_structure = ""
    for f, f_type in file_pairs:
        repo_structure += f"{f} - {f_type}\n"
    return repo_structure

class ObservationResult(BaseModel):
    Analysis: Optional[str]=Field(description="Analyzing the goal, repository file structure and intermediate output.")
    FinalAnswer: Optional[str]=Field(description="the final answer for the goal")
    Thoughts: Optional[str]=Field(description="If the information is insufficient, the thoughts will be given and be taken into consideration in next round.")

def convert_plan_to_string(plan: PlanAgentResult) -> str:
    plan_str = ""
    for action in plan.actions:
        action_str = f"Step: {action['name']}\n"
        action_str += f"Step Input: {action['input']}\n"
        plan_str += action_str
    return plan_str

STRING_TO_OBJECT_SYSTEM_PROMPT = """
You are an expert to understand data. You will be provided a text, and your task is to extracted structured data from the provided text.

---

### **Instructions**
1. If no structured data can be extracted, return None

---

### **Input Text**
{input_text}
"""

def try_parse_json_object(json_obj: str) -> dict | None:
    json_obj = json_obj.strip()

    # First, try to parse
    try:
        obj = json.loads(json_obj)
        return obj
    except JSONDecodeError as e:
        logger.error(e)

    # Second, let's handle some common errors
    if not json_obj.startswith("{") and not json_obj.endswith("}") and ":" in json_obj:
        json_obj = "{" + json_obj + "}"
    if json_obj.startswith("{{"):
        json_obj = json_obj[1:]
    if json_obj.endswith("}}"):
        json_obj = json_obj[:-1]

    # Finally, let's try to parse again
    try:
        obj = json.loads(json_obj)
        return obj
    except JSONDecodeError as e:
        logger.error(e)
        return None
    except Exception as e:
        logger.error(e)
        return None
    
def try_parse_with_llm(llm: BaseChatOpenAI, input_text: str, schema: any):
    system_prompt = ChatPromptTemplate.from_template(
        STRING_TO_OBJECT_SYSTEM_PROMPT
    ).format(input_text=input_text)
    
    conversation = CommonConversation(llm=llm)
    res, token_usage = conversation.generate_with_schema(
        system_prompt=system_prompt,
        instruction_prompt="Let's start to parse the input text.",
        schema=schema,
    )
    return res, token_usage

def read_license_file(repo_path: str) -> tuple[str | None, str|None]:
    # find hardcoded license file
    hardcoded_license_files = [
        "LICENSE",
        "LICENSE.txt",
        "LICENSE.md",
        "LICENSE.rst",
    ]
    license_files = []
    for file in hardcoded_license_files:
        file_path = os.path.join(str(repo_path), file)
        file_path = file_path.strip()
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                license_files.append((f.read(), os.path.join(repo_path, file)))
    
    max_item = max(license_files, key=lambda x: len(x[0])) if len(license_files) > 0 else (None, None)
    if max_item[0] is not None:
        return max_item[0], max_item[1]

    # find in root directory
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.lower() == "license":
                with open(os.path.join(root, file), "r") as f:
                    return f.read(), os.path.join(root, file)
            if file[:8].lower() == "license.":
                with open(os.path.join(root, file), "r") as f:
                    return f.read(), os.path.join(root, file)
    return None, None

