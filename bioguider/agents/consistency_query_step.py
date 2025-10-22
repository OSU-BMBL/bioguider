

from bioguider.agents.common_step import CommonStep
from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE


class ConsistencyQueryStep(CommonStep):
    def __init__(self, code_structure_db: CodeStructureDb):
        super().__init__()
        self.step_name = "Consistency Query Step"
        self.code_structure_db = code_structure_db

    def _execute_directly(self, state: ConsistencyEvaluationState):
        functions_and_classes = state["functions_and_classes"]
        all_rows: list[any] = []
        for function_or_class in functions_and_classes:
            function_or_class_name = function_or_class["name"] if "name" in function_or_class else "N/A"
            function_or_class_file_path = function_or_class["file_path"] if "file_path" in function_or_class else "N/A"
            function_or_class_parameters = function_or_class["parameters"] if "parameters" in function_or_class else "N/A"
            function_or_class_parent = function_or_class["parent"] if "parent" in function_or_class else "N/A"
            self._print_step(state, step_output=(
                f"Consistency Query Step: \n{function_or_class_name},\n"
                f" {function_or_class_file_path},\n"
                f" {function_or_class_parameters},\n"
                f" {function_or_class_parent}"
            ))
            file_path = None
            parent = None
            name = None
            if "file_path" in function_or_class and function_or_class["file_path"] != "N/A":
                file_path = function_or_class["file_path"]            
            if "parent" in function_or_class and function_or_class["parent"] != "N/A":
                parent = function_or_class["parent"]
            if "name" in function_or_class and function_or_class["name"] != "N/A":
                name = function_or_class["name"]
            
            rows: list[any] | None = None
            if name is None:
                if file_path is not None:
                    rows = self.code_structure_db.select_by_path(file_path)
                elif parent is not None:
                    rows = self.code_structure_db.select_by_parent(parent)
            else:
                if file_path is not None and parent is not None:
                    rows = self.code_structure_db.select_by_name_and_parent_and_path(name, parent, file_path)
                    rows = rows if rows is None else [rows]
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name_and_path(name, file_path)
                        rows = rows if rows is None else [rows]
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name_and_parent(name, parent)
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name(name)
                elif file_path is not None:
                    rows = self.code_structure_db.select_by_name_and_path(name, file_path)
                    rows = rows if rows is None else [rows]
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name(name)
                elif parent is not None:
                    rows = self.code_structure_db.select_by_name_and_parent(name, parent)
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name(name)
                else:
                    rows = self.code_structure_db.select_by_name(name)
            if rows is None or len(rows) == 0:
                self._print_step(state, step_output=f"No such function or class {name}")
                continue
            all_rows.extend(rows)

        state["all_query_rows"] = all_rows

        return state, {**DEFAULT_TOKEN_USAGE}
           
            

