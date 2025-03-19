import os
from dotenv import load_dotenv
import logging

from bioguider.agents.agent_utils import get_llm

from bioguider.agents.item_evaluation_agent import ItemEvaluationAgent
from bioguider.agents.prompt_utils import EVALUATION_ITEMS

load_dotenv()

logger = logging.getLogger(__name__)

def test_ItemEvaluationAgent():
    llm = get_llm(api_key=os.getenv("DEEPSEEK_API_KEY"), model_name="deepseek-chat")
    agent = ItemEvaluationAgent()
    agent.compile()
    res = agent.go(
        llm=llm, 
        evaluation_item=EVALUATION_ITEMS[1][0],
        score_point=EVALUATION_ITEMS[1][1],
        repo_path="/home/ubuntu/projects/github/tabula-data"
    )
    logger.info(f"Criterion **{EVALUATION_ITEMS[0][0]}**: {res[0]}")
    logger.info(f"Thoughts: {res[1]}")
    assert len(res) == 2
    assert (type(res[0]) == float or type(res[0]) == int) and res[0] > 0
    assert type(res[1]) == str and len(res[1]) > 0