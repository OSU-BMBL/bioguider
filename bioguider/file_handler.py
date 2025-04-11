# FileHandler 类，实现对文件的读写操作，这里的文件包括markdown文件和python文件
# repo_agent/file_handler.py
import ast
import json
import os

# import git
# from colorama import Fore, Style
from tqdm import tqdm

# from repo_agent.log import logger
from bioguider.settings import SettingsManager
from bioguider.utils.gitignore_checker import GitignoreChecker
# from repo_agent.utils.meta_info_utils import latest_verison_substring


class FileHandler:
    """
    历变更后的文件的循环中，为每个变更后文件（也就是当前文件）创建一个实例
    """

    def __init__(self, repo_path, file_path):
        self.file_path = file_path  # 这里的file_path是相对于仓库根目录的路径
        self.repo_path = repo_path

        setting = SettingsManager.get_setting()

        self.project_hierarchy = (
            setting.project.target_repo / setting.project.hierarchy_name
        )

    def read_file(self):
        """
        Read the file content

        Returns:
            str: The content of the current changed file
        """
        abs_file_path = os.path.join(self.repo_path, self.file_path)

        with open(abs_file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def get_obj_code_info(
        self, code_type, code_name, start_line, end_line, params, file_path=None
    ):
        """
        Get the code information for a given object.

        Args:
            code_type (str): The type of the code.
            code_name (str): The name of the code.
            start_line (int): The starting line number of the code.
            end_line (int): The ending line number of the code.
            parent (str): The parent of the code.
            file_path (str, optional): The file path. Defaults to None.

        Returns:
            dict: A dictionary containing the code information.
        """

        code_info = {}
        code_info["type"] = code_type
        code_info["name"] = code_name
        code_info["md_content"] = []
        code_info["code_start_line"] = start_line
        code_info["code_end_line"] = end_line
        code_info["params"] = params

        with open(
            os.path.join(
                self.repo_path, file_path if file_path != None else self.file_path
            ),
            "r",
            encoding="utf-8",
        ) as code_file:
            lines = code_file.readlines()
            code_content = "".join(lines[start_line - 1 : end_line])
            # 获取对象名称在第一行代码中的位置
            name_column = lines[start_line - 1].find(code_name)
            # 判断代码中是否有return字样
            if "return" in code_content:
                have_return = True
            else:
                have_return = False

            code_info["have_return"] = have_return
            # # 使用 json.dumps 来转义字符串，并去掉首尾的引号
            # code_info['code_content'] = json.dumps(code_content)[1:-1]
            code_info["code_content"] = code_content
            code_info["name_column"] = name_column

        return code_info

    def write_file(self, file_path, content):
        """
        Write content to a file.

        Args:
            file_path (str): The relative path of the file.
            content (str): The content to be written to the file.
        """
        # 确保file_path是相对路径
        if file_path.startswith("/"):
            # 移除开头的 '/'
            file_path = file_path[1:]

        abs_file_path = os.path.join(self.repo_path, file_path)
        os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
        with open(abs_file_path, "w", encoding="utf-8") as file:
            file.write(content)

    def get_end_lineno(self, node):
        """
        Get the end line number of a given node.

        Args:
            node: The node for which to find the end line number.

        Returns:
            int: The end line number of the node. Returns -1 if the node does not have a line number.
        """
        if not hasattr(node, "lineno"):
            return -1  # 返回-1表示此节点没有行号

        end_lineno = node.lineno
        for child in ast.iter_child_nodes(node):
            child_end = getattr(child, "end_lineno", None) or self.get_end_lineno(child)
            if child_end > -1:  # 只更新当子节点有有效行号时
                end_lineno = max(end_lineno, child_end)
        return end_lineno

    def add_parent_references(self, node, parent=None):
        """
        Adds a parent reference to each node in the AST.

        Args:
            node: The current node in the AST.

        Returns:
            None
        """
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self.add_parent_references(child, node)

    def get_functions_and_classes(self, code_content):
        """
        Retrieves all functions, classes, their parameters (if any), and their hierarchical relationships.
        Output Examples: [('FunctionDef', 'AI_give_params', 86, 95, None, ['param1', 'param2']), ('ClassDef', 'PipelineEngine', 97, 104, None, []), ('FunctionDef', 'get_all_pys', 99, 104, 'PipelineEngine', ['param1'])]
        On the example above, PipelineEngine is the Father structure for get_all_pys.

        Args:
            code_content: The code content of the whole file to be parsed.

        Returns:
            A list of tuples containing the type of the node (FunctionDef, ClassDef, AsyncFunctionDef),
            the name of the node, the starting line number, the ending line number, the name of the parent node, and a list of parameters (if any).
        """
        tree = ast.parse(code_content)
        self.add_parent_references(tree)
        functions_and_classes = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                # if node.name == "recursive_check":
                #     import pdb; pdb.set_trace()
                start_line = node.lineno
                end_line = self.get_end_lineno(node)
                # def get_recursive_parent_name(node):
                #     now = node
                #     while "parent" in dir(now):
                #         if isinstance(now.parent, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                #             assert 'name' in dir(now.parent)
                #             return now.parent.name
                #         now = now.parent
                #     return None
                # parent_name = get_recursive_parent_name(node)
                parameters = (
                    [arg.arg for arg in node.args.args] if "args" in dir(node) else []
                )
                all_names = [item[1] for item in functions_and_classes]
                # (parent_name == None or parent_name in all_names) and
                functions_and_classes.append(
                    (type(node).__name__, node.name, start_line, end_line, parameters)
                )
        return functions_and_classes

    def generate_file_structure(self, file_path):
        """
        Generates the file structure for the given file path.

        Args:
            file_path (str): The relative path of the file.

        Returns:
            dict: A dictionary containing the file path and the generated file structure.

        Output example:
        {
            "function_name": {
                "type": "function",
                "start_line": 10,
                ··· ···
                "end_line": 20,
                "parent": "class_name"
            },
            "class_name": {
                "type": "class",
                "start_line": 5,
                ··· ···
                "end_line": 25,
                "parent": None
            }
        }
        """
        with open(os.path.join(self.repo_path, file_path), "r", encoding="utf-8") as f:
            content = f.read()
            structures = self.get_functions_and_classes(content)
            file_objects = []  # 以列表的形式存储
            for struct in structures:
                structure_type, name, start_line, end_line, params = struct
                code_info = self.get_obj_code_info(
                    structure_type, name, start_line, end_line, params, file_path
                )
                file_objects.append(code_info)

        return file_objects

    def generate_overall_structure(self, file_path_reflections, jump_files) -> dict:
        """获取目标仓库的文件情况，通过AST-walk获取所有对象等情况。
        对于jump_files: 不会parse，当做不存在
        """
        repo_structure = {}
        gitignore_checker = GitignoreChecker(
            directory=self.repo_path,
            gitignore_path=os.path.join(self.repo_path, ".gitignore"),
        )

        bar = (gitignore_checker.check_files_and_folders())
        for not_ignored_files in bar:
            normal_file_names = not_ignored_files
            if not_ignored_files in jump_files:
                print(
                    f"[File-Handler] Unstaged AddFile, ignore this file: {normal_file_names}"
                )
                continue
          
            try:
                repo_structure[normal_file_names] = self.generate_file_structure(
                    not_ignored_files
                )
            except Exception as e:
                print(
                    f"Alert: An error occurred while generating file structure for {not_ignored_files}: {e}"
                )
                continue
            # bar.set_description(f"generating repo structure: {not_ignored_files}")
        return repo_structure

    def convert_to_markdown_file(self, file_path=None):
        """
        Converts the content of a file to markdown format.

        Args:
            file_path (str, optional): The relative path of the file to be converted. If not provided, the default file path, which is None, will be used.

        Returns:
            str: The content of the file in markdown format.

        Raises:
            ValueError: If no file object is found for the specified file path in project_hierarchy.json.
        """
        with open(self.project_hierarchy, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if file_path is None:
            file_path = self.file_path

        # Find the file object in json_data that matches file_path

        file_dict = json_data.get(file_path)

        if file_dict is None:
            raise ValueError(
                f"No file object found for {self.file_path} in project_hierarchy.json"
            )

        markdown = ""
        parent_dict = {}
        objects = sorted(file_dict.values(), key=lambda obj: obj["code_start_line"])
        for obj in objects:
            if obj["parent"] is not None:
                parent_dict[obj["name"]] = obj["parent"]
        current_parent = None
        for obj in objects:
            level = 1
            parent = obj["parent"]
            while parent is not None:
                level += 1
                parent = parent_dict.get(parent)
            if level == 1 and current_parent is not None:
                markdown += "***\n"
            current_parent = obj["name"]
            params_str = ""
            if obj["type"] in ["FunctionDef", "AsyncFunctionDef"]:
                params_str = "()"
                if obj["params"]:
                    params_str = f"({', '.join(obj['params'])})"
            markdown += f"{'#' * level} {obj['type']} {obj['name']}{params_str}:\n"
            markdown += (
                f"{obj['md_content'][-1] if len(obj['md_content']) >0 else ''}\n"
            )
        markdown += "***\n"

        return markdown


if __name__ == "__main__":
    # fh = FileHandler(
    #     repo_path="/home/ubuntu/projects/github/tabula-data",
    #     file_path="/home/ubuntu/projects/github/tabula-data/app.py",
    # )
    # res = fh.generate_overall_structure("", [])
    checker = GitignoreChecker(
        directory="/home/ubuntu/projects/github/tabula-data",
        gitignore_path="/home/ubuntu/projects/github/tabula-data/.gitignore",
    )
    res = checker.check_files_and_folders()
    print(res)