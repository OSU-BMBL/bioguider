import ast
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CLIArgument:
    """A single ``argparse`` argument definition extracted statically from source."""
    option_strings: list[str]          # ["-f", "--file"]; empty list => positional
    dest: str                          # "file"
    arg_type: Optional[str] = None     # "int", "str", "Path", ... (source text of the callable)
    default: Optional[str] = None      # str(value) for a literal, ast.unparse() otherwise, "<dynamic>" if unknown
    action: Optional[str] = None       # "store_true", "store_false", "append", "count", ...
    nargs: Optional[str] = None        # "?", "+", "*", or an int rendered as str
    choices: Optional[list[str]] = None
    required: Optional[bool] = None
    metavar: Optional[str] = None
    help: Optional[str] = None
    subcommand: Optional[str] = None   # name of the sub-parser this belongs to, if any
    lineno: int = -1


@dataclass
class CLIInterface:
    """Static view of a Python module's ``argparse`` command-line interface."""
    prog: Optional[str] = None
    description: Optional[str] = None
    arguments: list[CLIArgument] = field(default_factory=list)
    subcommands: list[str] = field(default_factory=list)


def _literal_or_source(node: Optional[ast.AST]) -> Any:
    """Return the literal value of *node* if it is a constant expression,
    otherwise its source text (``ast.unparse``), or ``"<dynamic>"`` on failure."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        pass
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - extremely defensive
        return "<dynamic>"


def _as_opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


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

    # ------------------------------------------------------------------ #
    # argparse command-line-interface extraction
    # ------------------------------------------------------------------ #

    def get_cli_interface(self) -> Optional[CLIInterface]:
        """
        Statically extract the ``argparse`` command-line interface of this file.

        Walks the AST (no code is executed) looking for:
          - ``argparse.ArgumentParser(...)`` / ``ArgumentParser(...)`` instantiations
            (for ``prog`` / ``description``),
          - ``<parser>.add_subparsers()`` + ``<subparsers>.add_parser("name", ...)``
            (for sub-command names and per-sub-command argument grouping),
          - ``<parser>.add_argument(...)`` calls (the actual options/positionals).

        Returns a :class:`CLIInterface`, or ``None`` if the file does not look
        like it builds an ``argparse`` CLI.

        Note: only ``argparse`` is recognised; ``click`` / ``typer`` / ``docopt``
        are out of scope for now.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError):
            return None

        iface = CLIInterface()
        saw_argparse = False

        # Map a variable name to the sub-command it refers to, e.g.
        #   sub = subparsers.add_parser("train")  ->  {"sub": "train"}
        var_to_subcommand: dict[str, str] = {}

        # First pass: ArgumentParser(...) instantiations and add_parser(...) assignments.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and self._is_argument_parser_call(node.func):
                saw_argparse = True
                prog = self._kw_value(node, "prog")
                description = self._kw_value(node, "description")
                if prog is not None and iface.prog is None:
                    iface.prog = _as_opt_str(prog)
                if description is not None and iface.description is None:
                    iface.description = _as_opt_str(description)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_parser"
            ):
                saw_argparse = True
                name = None
                if node.args:
                    name = _literal_or_source(node.args[0])
                if name is not None:
                    name = str(name)
                    if name not in iface.subcommands:
                        iface.subcommands.append(name)
                    # Bind any assignment target(s) of this call to the sub-command name.
                    for assign in ast.walk(tree):
                        if isinstance(assign, ast.Assign) and assign.value is node:
                            for tgt in assign.targets:
                                if isinstance(tgt, ast.Name):
                                    var_to_subcommand[tgt.id] = name

        # Second pass: add_argument(...) calls.
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            saw_argparse = True
            arg = self._parse_add_argument(node)
            if arg is None:
                continue
            if isinstance(node.func.value, ast.Name):
                arg.subcommand = var_to_subcommand.get(node.func.value.id)
            iface.arguments.append(arg)

        if not saw_argparse:
            return None
        return iface

    def get_cli_arguments(self) -> list[tuple]:
        """
        Flat tuple view of :meth:`get_cli_interface`, convenient for DB insertion.

        Each tuple is::

            (path, prog, description, subcommand, dest, option_strings, arg_type,
             default, action, nargs, choices, required, metavar, help, lineno)

        ``option_strings`` and ``choices`` are Python lists (the caller is
        expected to serialise them).  Returns ``[]`` when there is no CLI.
        """
        iface = self.get_cli_interface()
        if iface is None:
            return []
        path = str(self.file_path)
        rows: list[tuple] = []
        for a in iface.arguments:
            rows.append((
                path, iface.prog, iface.description, a.subcommand, a.dest,
                a.option_strings, a.arg_type, a.default, a.action, a.nargs,
                a.choices, a.required, a.metavar, a.help, a.lineno,
            ))
        return rows

    # ------------------------------------------------------------------ #
    # argparse parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_argument_parser_call(func: ast.AST) -> bool:
        """True if *func* is ``ArgumentParser`` or ``argparse.ArgumentParser``."""
        if isinstance(func, ast.Name):
            return func.id == "ArgumentParser"
        if isinstance(func, ast.Attribute):
            return func.attr == "ArgumentParser"
        return False

    @staticmethod
    def _kw_value(call: ast.Call, name: str) -> Any:
        for kw in call.keywords:
            if kw.arg == name:
                return _literal_or_source(kw.value)
        return None

    @classmethod
    def _parse_add_argument(cls, call: ast.Call) -> Optional[CLIArgument]:
        # Positional name args: "-f", "--file" (options) or "input" (positional).
        name_args: list[str] = []
        for a in call.args:
            val = _literal_or_source(a)
            if isinstance(val, str):
                name_args.append(val)
        option_strings = [n for n in name_args if n.startswith("-")]
        positional_name = None
        if not option_strings and name_args:
            positional_name = name_args[0]

        kw = {k.arg: k.value for k in call.keywords if k.arg is not None}

        explicit_dest = _literal_or_source(kw["dest"]) if "dest" in kw else None
        if isinstance(explicit_dest, str):
            dest = explicit_dest
        elif option_strings:
            longs = [o for o in option_strings if o.startswith("--")]
            chosen = longs[0] if longs else option_strings[0]
            dest = chosen.lstrip("-").replace("-", "_")
        elif positional_name:
            dest = positional_name.lstrip("-").replace("-", "_")
        else:
            return None  # nothing usable

        arg_type = ast.unparse(kw["type"]) if "type" in kw else None

        default_val = _literal_or_source(kw["default"]) if "default" in kw else None
        action_val = _literal_or_source(kw["action"]) if "action" in kw else None
        nargs_val = _literal_or_source(kw["nargs"]) if "nargs" in kw else None
        metavar_val = _literal_or_source(kw["metavar"]) if "metavar" in kw else None
        help_val = _literal_or_source(kw["help"]) if "help" in kw else None
        required_val = _literal_or_source(kw["required"]) if "required" in kw else None

        choices_val: Optional[list[str]] = None
        if "choices" in kw:
            raw = _literal_or_source(kw["choices"])
            if isinstance(raw, (list, tuple, set)):
                choices_val = [str(x) for x in raw]
            elif raw is not None:
                choices_val = [str(raw)]

        return CLIArgument(
            option_strings=option_strings,
            dest=dest,
            arg_type=_as_opt_str(arg_type),
            default=_as_opt_str(default_val),
            action=_as_opt_str(action_val),
            nargs=_as_opt_str(nargs_val),
            choices=choices_val,
            required=bool(required_val) if isinstance(required_val, bool) else None,
            metavar=_as_opt_str(metavar_val),
            help=_as_opt_str(help_val),
            lineno=getattr(call, "lineno", -1),
        )

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
