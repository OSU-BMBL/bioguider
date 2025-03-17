
from .item_evaluation_agent import ItemEvaluationAgent
from .prompt_utils import EVALUATION_ITEMS

class EvaluationAgent:
    def __init__(self):
        pass

    def evaluate(self, llm, repo_path: str):
        self.llm = llm

        report = ""
        for item in EVALUATION_ITEMS:
            item_eval_agent = ItemEvaluationAgent()
            item_eval_agent.compile()
            res = item_eval_agent.go(
                llm=self.llm,
                evaluation_item=item[0],
                score_point=item[1],
                repo_path=repo_path,
            )
            report += f"\nEvaluation Item: {item[0]}\n\nscore: {res[0]}\n\nthoughts: {res[1]}\n\n"

        return report

