import os
from typing import List, Literal, Optional, TypedDict, Union
import logging
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate

from .prompt_utils import EVALUATION_ITEM_PROMPT, EVALUATION_SYSTEM_PROMPT
from .agent_utils import read_file_or_dir

logger = logging.getLogger(__name__)

class EvaluationItemResult(BaseModel):
    """ Item evaluation result schema"""
    evaluation_item: Optional[str] = Field(description="the name of evaluation item, like 'Clarity & Readability' or 'Completeness'")
    Score: Optional[float] = Field(description="Score out of 20")
    Reason: Optional[str] = Field(description="Reasoning for the score")
    files_or_directories: Optional[List[str]] = Field(
        description="List of files or directories needed for evaluation"
    )

class ItemEvaluationState(TypedDict):
  """ Langgraph state data, 
  This class represents the state data for Langgraph, which will be passed between states during execution"""
  files_or_directories: List[str]
  Score: Optional[float]
  Reason: Optional[str]

class ItemEvaluationAgent:
    """
    Item evaluation agent
    This class is to evaluate specified repo on a evaluation criterion.
    """
    def __init__(self):
        self.llm = None
        self.graph: StateGraph = None
        self.evaluation_item: Union[str, None] = None
        self.score_point: Union[float, None] = None
        self.repo_pth: Union[str, None] = None
        self.system_prompt: Union[str, None] = None

        self.runtime_files_or_directories: list[str] = []
        self.runtime_files_or_directories_content: list[str] = []

    def compile(self):
        def evaluate(state: ItemEvaluationState):

            system_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("placeholder", "{messages}"),
            ])
            contents = self.runtime_files_or_directories_content
            content = "\n".join(contents) if len(contents) > 0 else "None\n"
            item_prompt = EVALUATION_ITEM_PROMPT.format(
                files_or_directories=content,
                evaluation_item=self.evaluation_item,
                score_point=self.score_point,
            )

            state["files_or_directories"] = None
            agent = system_prompt | self.llm.with_structured_output(EvaluationItemResult)
            res = agent.invoke(input={"messages": [(
                "user", item_prompt
            )]})
            if res.Score is not None:
              state["Score"] = res.Score
              state["Reason"] = res.Reason
              state["files_or_directories"] = []
            else:
              state["files_or_directories"] = res.files_or_directories
            return state
        
        def tool_read_file_or_dir(state: ItemEvaluationState):
            if state["files_or_directories"] is None or\
                len(state["files_or_directories"]) == 0:
                return state
            files_to_provide = state["files_or_directories"]
            files = [f for f in files_to_provide if not f in self.runtime_files_or_directories]
            contents = []
            try_not_summarize = True if len(state["files_or_directories"]) < 2 else False
            for f in files:
                fname = os.path.join(self.repo_pth, f)  
                file_content = read_file_or_dir(
                    fname,
                    repo_path=self.repo_pth, 
                    gitignore_path=f'{self.repo_pth}/.gitignore',
                    llm=self.llm,
                    try_not_summarize=try_not_summarize
                )
                contents.append(f"{fname}: \n```{file_content}```")
            self.runtime_files_or_directories += files
            self.runtime_files_or_directories_content += contents
            return state

        def evaluate_status(state: ItemEvaluationState) -> Literal["evaluate_item", "__end__"]:
            if "Score" in state and state["Score"] is not None:
                return "__end__"
            return "evaluate_item"
        
        graph = StateGraph(ItemEvaluationState)
        graph.add_node("evaluate_item", evaluate)
        graph.add_node("read_files_or_directories", tool_read_file_or_dir)
        graph.add_edge(START, "evaluate_item")
        graph.add_edge("evaluate_item", "read_files_or_directories")
        graph.add_conditional_edges(
            "read_files_or_directories", evaluate_status, "evaluate_item"
        )
        self.graph = graph.compile()
        # display(Image(self.graph.get_graph().draw_mermaid_png()))            
        logger.info(self.graph.get_graph().draw_ascii())

    def _initialize(
        self,
        llm,
        evaluation_item: str,
        score_point: float,
        repo_path: str
    ):
        self.llm = llm
        self.evaluation_item = evaluation_item
        self.score_point = score_point
        self.repo_pth = repo_path
        self.runtime_files_or_directories = []
        self.runtime_files_or_directories_content = []

    def go(self, 
           llm,
           evaluation_item: str,
           score_point: float,
           repo_path: str):
        # initialize
        self._initialize(llm, evaluation_item, score_point, repo_path)

        repo_stucture = read_file_or_dir(
            repo_path, repo_path, os.path.join(repo_path, ".gitignore"), llm
        )
        self.system_prompt = EVALUATION_SYSTEM_PROMPT.format(repository_structure=repo_stucture)

        config = {"recursion_limit": 500}
        for s in self.graph.stream(
            {"files_or_directories": []}, 
            config=config, 
            stream_mode="values",
        ):
            if "Score" in s and s["Score"] is not None and \
                "Reason" in s and s["Reason"] is not None:
                return (s["Score"], s["Reason"])
            logger.info(s)

