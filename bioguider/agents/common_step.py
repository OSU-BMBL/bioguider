from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import logging

from bioguider.agents.workflow_utils import BGWorkflowState
from bioguider.agents.prompt_utils import USER_INSTRUCTION
from bioguider.agents.workflow_utils import (
    bg_enter_step,
    bg_leave_step,
)
from bioguider.agents.common_agent import (
    CommonAgentResult,
    CommonAgent,
)

logger = logging.getLogger(__name__)


class CommonStep(ABC):
    def __init__(self):
        super().__init__()
        self.start_title = ""
        self.start_descp = ""
        self.end_title = ""

    def enter_step(self, state: BGWorkflowState):
        bg_enter_step(state, self.start_title, self.start_descp)

    def leave_step(
        self,
        state: BGWorkflowState,
        res: CommonAgentResult | None = None,
        processed_res: Any | None = None,
        token_usage: dict | None = None,
    ):
        bg_leave_step(
            state=state,
            step_output=self.end_title,
            step_reasoning_process=res["reasoning_process"]
            if res is not None and "reasoning_process" in res
            else None,
            token_usage=token_usage,
        )

    def _step_output(
        self,
        state: BGWorkflowState,
        step_output: Optional[str] = None,
        step_reasoning_process: Optional[str] = None,
    ):
        def default_output(
            step_output: Optional[str] = None,
            step_reasoning_process: Optional[str] = None,
        ):
            if step_reasoning_process is not None:
                logger.info(f"\n\nReasoning: \n{step_reasoning_process}\n\n")
            if step_output is not None:
                logger.info(step_output)

        step_callback = (
            state["step_callback"]
            if "step_callback" in state and state["step_callback"] is not None
            else default_output
        )
        step_callback(
            step_reasoning_process=step_reasoning_process,
            step_output=step_output,
        )

    def execute(self, state: BGWorkflowState):
        self.enter_step(state)
        res, processed_res, token_usage = self.execute_directly(state)
        self.leave_step(state, res, processed_res, token_usage)

        return state

    @abstractmethod
    def execute_directly(
        self,
        state: BGWorkflowState,
    ) -> tuple[CommonAgentResult, Any | None, dict | None]:
        """execute directly"""


class CommonAgentStep(CommonStep):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_system_prompt(self, state: BGWorkflowState):
        """get system prompt"""

    def get_instruction_prompt(self, state: BGWorkflowState):
        """get instruction prompt"""
        return USER_INSTRUCTION

    @abstractmethod
    def get_schema(self) -> CommonAgentResult | dict:
        """get result schema (pydantic BaseModel or json schema)"""

    @abstractmethod
    def get_post_processor_and_kwargs(
        self, state: BGWorkflowState
    ) -> tuple[
        Callable | None,
        dict | None,
    ]:
        """get post_processor and its kwargs"""

    def execute_directly(
        self, state: BGWorkflowState
    ) -> tuple[
        CommonAgentResult,
        Any | None,
        dict | None,
    ]:
        """execute step directly"""
        system_prompt = self.get_system_prompt(state)
        instruction_prompt = self.get_instruction_prompt(state)
        llm = state["llm"]
        schema = self.get_schema()
        post_process, kwargs = self.get_post_processor_and_kwargs(state)
        agent = CommonAgent(llm=llm)
        if kwargs is not None:
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process,
                **kwargs,
            )
        else:
            res, processed_res, token_usage = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=instruction_prompt,
                schema=schema,
                post_process=post_process,
            )
        reasoning_process = ""
        if res is not None:
            try:
                reasoning_process = (
                    res["reasoning_process"]
                    if type(res) == dict
                    else res.reasoning_process
                )
            except Exception as e:
                logger.error(
                    f"Failed to access res.reasoning_process.\nError is {str(e)}"
                )
                pass
        self._step_output(state, step_reasoning_process=reasoning_process)
        return res, processed_res, token_usage