
from pydantic import PrivateAttr
import re
import io
import contextlib
import logging
from langchain_experimental.tools.python.tool import PythonAstREPLTool

class CustomPythonAstREPLTool(PythonAstREPLTool):
    """
    Custom Python REPL tool that executes Python code and captures output.
    This tool is designed to be used in a LangChain agent for executing Python code
    and capturing the output, including any print statements.
    """
    __name__ = "Custom_Python_AST_REPL"
    _exec_globals: dict = PrivateAttr()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exec_globals = {}
        self._exec_globals.update(__builtins__)

    def _set_globals(self, table_dict=None):
        self._exec_globals = {}
        self._exec_globals.update(__builtins__)

        if table_dict is not None:
            self._exec_globals.update(table_dict)

    def _run(self, query: str, run_manager=None):  
        print("================================== code here ==============================")
        print(query)
        print("===========================================================================")
        code_match = re.search(r"```(.*?)```", query, re.DOTALL)
        if code_match:
            # Extract code within backticks
            code = code_match.group(1)
        else:
            code = query
        code = code.strip()
        if code.startswith("python"):
            code = code[len("python"):].lstrip()
        
        if code.endswith("Observation"):
            code = code[:-len("Observation")].rstrip()
        
        code_lines = code.strip().split('\n')
        code = '\n'.join(code_lines[:-1])   # avoid printing the last line twice
        last_line = code_lines[-1]
        
        output_capture = io.StringIO()
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            logging.getLogger().handlers[0].stream = output_capture
            try:
                exec(code, self._exec_globals)
                try:
                    result = eval(last_line, self._exec_globals)
                    if result is not None:
                        print(result, file=output_capture)
                except:
                    pass
            except Exception as e:
                return str(e)
        
        # Retrieve the output and return it
        output = output_capture.getvalue()
        return output if output else "Execution completed without output."



