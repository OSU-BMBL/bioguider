import ast
import os

class PythonFileHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_functions_and_classes(self) -> list[str]:
        """
        Get the functions and classes in a given file.
        Returns a list of tuples, each containing:
        1. the function or class name,
        2. parent name,
        3. start line number,
        4. end line number,
        5. doc string,
        6. params.
        """
        with open(self.file_path, 'r') as f:
            tree = ast.parse(f.read())
            functions_and_classes = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                    start_lineno = node.lineno
                    end_lineno = self.get_end_lineno(node)
                    doc_string = ast.get_docstring(node)
                    params = (
                        [arg.arg for arg in node.args.args] if "args" in dir(node) else []
                    )
                    parent = None
                    functions_and_classes.append((node.name, parent, start_lineno, end_lineno, doc_string, params))
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            start_lineno = child.lineno
                            end_lineno = self.get_end_lineno(child)
                            doc_string = ast.get_docstring(child)
                            params = (
                                [arg.arg for arg in child.args.args] if "args" in dir(child) else []
                            )
                            parent = node.name
                            functions_and_classes.append((child.name, parent, start_lineno, end_lineno, doc_string, params))
            return functions_and_classes
    
    def get_imports(self) -> list[str]:
        pass

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