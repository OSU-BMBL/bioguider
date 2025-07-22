
from typing import Optional
from pydantic import BaseModel, Field
from bioguider.agents.agent_utils import try_parse_json_object, try_parse_with_llm
from bioguider.agents.evaluation_task import EvaluationTask
from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.identification_task import IdentificationTask
from bioguider.agents.prompt_utils import CollectionGoalItemEnum
from bioguider.agents.evaluation_installation_task import StructuredEvaluationInstallationResult
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE

DEMO_INSTRUCTION_GOAL = """
1. Identify if it provides the instructions to run on provided data
2. Identify if it provides the instructions to run on custom data
3. Identify if it provides the expected output
"""

DEMO_INSTRUCTION_FINAL_ANSWER = \
'{{"run_on_data_instruction": <True or False>, "run_on_custom_instruction": <True or False>, "expected_output_description": <True Or False>}}'

class DemoInstructionsResult(BaseModel):
    run_on_data_instruction: Optional[bool] = Field(description="A boolean value. Does it provide instructions on how to run on provided data?")
    run_on_custom_instruction: Optional[bool] = Field(description="A boolean value. Does it provide instructions on how to run on custom data?")
    expected_output_description: Optional[bool] = Field(description="A boolean value. Does it provide the description of expected output?")

class EvaluationSubmissionRequirementsTask(EvaluationTask):
    def __init__(
        self, 
        llm, 
        repo_path, 
        gitignore_path, 
        meta_data = None, 
        step_callback = None, 
        summarized_files_db = None,
        readme_files_evaluation: dict | None = None,
        installation_evaluation: dict | None = None,
        installation_files: list[str] | None = None
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback, summarized_files_db)
        self.evaluation_name = "Submission Requirements Evaluation"
        self.readme_files_evaluation = readme_files_evaluation
        self.installation_evaluation = installation_evaluation
        self.installation_files = installation_files

    def _collect_software_package_content(self):
        collection_task = CollectionTask(
            llm = self.llm,
            step_callback=self.step_callback,
            summarize_instruction="We are collecting compiled standalone files, source code files and example data files.",
            summarized_files_db=self.summarized_files_db,
        )
        collection_task.compile(
            repo_path=self.repo_path,
            gitignore_path=self.gitignore_path,
            db=self.summarized_files_db,
            goal_item=CollectionGoalItemEnum.SoftwarePackageContent.name,
        )
        files = collection_task.collect()

        return files
    
    def _evaluate_software_package_content(self):
        files = self._collect_software_package_content()
        if len(files) == 3:
            return {
                "compiled_standalone_software": files[0].strip().lower() != "n/a",
                "source_code": files[1].strip().lower() != "n/a",
                "demo_dataset": files[2].strip().lower() != "n/a",
            }, files
        else:
            return {
                "compiled_standalone_software": False,
                "source_code": False,
                "demo_dataset": False,
            }, files
    
    def _evaluatie_demo_instructions(self):
        readme_files = [f for f in self.readme_files_evaluation.keys() \
                        if self.readme_files_evaluation[f]["evaluation"]["project_level"]]
        installation_files = self.installation_files if self.installation_files is not None else []
        provided_files = readme_files + installation_files
        provided_files = provided_files if len(provided_files) > 0 else None
        identify_task = IdentificationTask(
            llm=self.llm,
            step_callback=self.step_callback,
            summarized_files_db=self.summarized_files_db,
            provided_files=provided_files
        )
        identify_task.compile(
            repo_path=self.repo_path,
            gitignore_path=self.gitignore_path,
        )
        final_answer = identify_task.identify_customize_goal(
            goal="demo instructions",
            final_answer_example=DEMO_INSTRUCTION_FINAL_ANSWER,
        )
        final_answer = final_answer["final_answer"] \
            if final_answer is not None and "final_answer" in final_answer else final_answer
        parsed_obj = self._parse_demo_instruction_result(final_answer)
        return parsed_obj, provided_files

    def _parse_demo_instruction_result(self, result: str | dict):
        if isinstance(result, dict):
            return result
        obj = try_parse_json_object(result)
        if obj is None:
            obj, token_usage = try_parse_with_llm(
                llm=self.llm,
                input_text=result,
                schema=DemoInstructionsResult,
            )
            obj = vars(obj) if obj is not None else obj
            self.print_step(token_usage=token_usage)
            self.print_step(step_output=str(obj))

        return obj


    def _combine_evaluation(
        self,
        software_evaluation: dict,
        demo_evaluation: dict,
    ):
        readme_files = [f for f in self.readme_files_evaluation.keys() \
                        if self.readme_files_evaluation[f]["evaluation"]["project_level"]]
        structured_install_evaluation: StructuredEvaluationInstallationResult = \
            self.installation_evaluation["structured_evaluation"]
        software_dependency = structured_install_evaluation.dependency_number > 0
        install_tutorial = structured_install_evaluation.install_tutorial
        license = any([self.readme_files_evaluation[f]["structured_evaluation"].license_score for f in readme_files])
        return {
            **software_evaluation,
            **demo_evaluation,
            "complete_readme": len(readme_files) > 0,
            "software_dependency": software_dependency,
            "install_tutorial": install_tutorial,
            "license": license,
        }

    def _evaluate(self, files):
        
        software_evaluation, software_files = self._evaluate_software_package_content()
        demo_evaluation, demo_files = self._evaluatie_demo_instructions()
        files = list(set(software_files + demo_files))

        return self._combine_evaluation(software_evaluation, demo_evaluation), {**DEFAULT_TOKEN_USAGE}, files


    def _collect_files(self):
        return []
        


