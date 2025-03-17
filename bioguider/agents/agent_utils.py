
import os
from typing import Union
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import logging

from .gitignore_checker import GitignoreChecker

logger = logging.getLogger(__name__)

def get_llm(
    api_key: str,
    model_name: str="gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 4096,
):
    if model_name.startswith("deepseek"):
        chat = ChatDeepSeek(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_name.startswith("gpt"):
        chat = ChatOpenAI(
            openai_api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError("Invalid model name")
    # validate chat
    try:
        chat.invoke("Hi")
    except Exception as e:
        print(e)
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

def read_file_or_dir(
    name: str, 
    repo_path: str, 
    gitignore_path: str,
    llm: any,
    try_not_summarize: bool = False
) -> Union[str, list]:
    if os.path.isfile(name):
        with open(name, 'r') as f:
            content = f.read()
        if try_not_summarize and len(content) < HUGE_FILE_LENGTH:
            return content
        else:
            return summarize_file(llm, name, content)
    if os.path.isdir(name):
        gitignore_checker = GitignoreChecker(
            directory=name,
            gitignore_path=gitignore_path,
        )
        not_ignored_files = gitignore_checker.check_files_and_folders(
            first_level=True
        )
        directory_content = "\n"
        for item in not_ignored_files:
            file_path = os.path.join(name, item)
            if os.path.isfile(file_path):
                directory_content += f"{file_path} - f\n"
            if os.path.isdir(file_path):
                directory_content += f"{file_path} - d\n"
        return directory_content
    return ""

EVALUATION_SUMMARIZE_FILE_PROMPT = ChatPromptTemplate.from_template(
"""
Here is the content of file {file_name}:
```
{file_content}
```

The content is lengthy, so please provide a concise summary in one to two sentences. Focus on assessing the quality and clarity of its documentation, highlighting key strengths or areas for improvement.
""")

MAX_FILE_LENGTH=1024 # 1K
def summarize_file(llm, name: str, content: str=None):
    if content is None:
        try:
            with open(name, "r") as fobj:
                content = fobj.read()
        except Exception as e:
            logger.error(e)
            return ""
    file_content = content
    if len(file_content) < MAX_FILE_LENGTH:
        return file_content
    
    file_content = content[:MAX_FILE_LENGTH] + " ..."
    app = create_react_agent(llm, [])
    prompt = EVALUATION_SUMMARIZE_FILE_PROMPT.format(file_name=name, file_content=file_content)
    config = {"recursion_limit": 500}
    inputs = {"messages": [("user", prompt)]}
    for s in app.stream(inputs, stream_mode="values", config=config):
        out = s['messages'][-1].content
    
    return out

    