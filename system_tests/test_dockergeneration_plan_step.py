import pytest
import os

from langchain.tools import StructuredTool, Tool
from bioguider.agents.dockergeneration_task_utils import (
    extract_python_file_from_notebook_tool,
    generate_Dockerfile_tool,
    prepare_provided_files_string,
    write_file_tool,
    DockerGenerationWorkflowState,
)
from bioguider.agents.dockergeneration_plan_step import (
    DockerGenerationPlanStep,
)
from bioguider.agents.agent_utils import (
    read_directory,
    read_file,
    generate_repo_structure_prompt,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool

def test_DockerGenerationPlanStep(llm, step_callback):
    repo_path = "/home/ubuntu/projects/github/Slide_recon"
    gitignore_path = "/home/ubuntu/projects/github/Slide_recon/.gitignore"
    files = read_directory(repo_path, gitignore_path)
    repo_structure = generate_repo_structure_prompt(files, repo_path)
    provided_files = [
        "README.md", "environment.yml", "demo/example.ipynb",
        "scripts/qsub_seq_blind.sh", "scripts/reconstruction.sh",
    ]
    str_provided_files = prepare_provided_files_string(repo_path, provided_files)
    write_tool = write_file_tool(repo_path)
    generate_tool = generate_Dockerfile_tool(
        llm=llm, 
        repo_path=repo_path, 
        extracted_files=str_provided_files, 
        repo_structure=repo_structure,
        output_callback=step_callback,
        plan_thoughts="N/A",
    )
    python_tool = CustomPythonAstREPLTool()
    extract_tool = extract_python_file_from_notebook_tool(repo_path)
    custom_tools = [
        StructuredTool.from_function(
            write_tool.run,
            description=write_tool.__class__.__doc__,
            name=write_tool.__class__.__name__,
        ),
        Tool(
            name=generate_tool.__class__.__name__,
            func=generate_tool.run,
            description=generate_tool.__class__.__doc__,
        ),
        python_tool,
        StructuredTool.from_function(
            extract_tool.run,
            description=extract_tool.__class__.__doc__,
            name=extract_tool.__class__.__name__,
        )
    ]

    step = DockerGenerationPlanStep(
        llm,
        repo_path,
        repo_structure,
        gitignore_path,
        custom_tools,
    )
    state = DockerGenerationWorkflowState(
        llm=llm,
        step_output_callback=step_callback,
        provided_files=provided_files,
        step_dockerfile_content=read_file(os.path.join(repo_path, "demo-bioguider-wvHOMqOxcZ.Dockerfile")),
        step_thoughts="""The runtime logs indicate an issue with the kernel specification for the notebook, named 'slidelock'. This kernel must be available in the environment, yet it is either not installed or referenced incorrectly. To resolve, verify that the environment.yml includes necessary dependencies for 'slidelock' or modify the notebook to rely on a universally supported kernel, such as 'python3'.""",
        step_output="""[NbConvertApp] Converting notebook example.ipynb to html
[NbConvertApp] WARNING | Kernelspec name slidelock cannot be found!
[NbConvertApp] ERROR | No such kernel named slidelock
Traceback (most recent call last):
  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 87, in wrapper
  out = await method(self, *args, **kwargs)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 435, in _async_start_kernel
  kernel_cmd, kw = await self._async_pre_start_kernel(**kw)
  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 397, in _async_pre_start_kernel
  self.kernel_spec,
  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 195, in kernel_spec
  self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/kernelspec.py", line 285, in get_kernel_spec
  raise NoSuchKernel(kernel_name)
  jupyter_client.kernelspec.NoSuchKernel: No such kernel named slidelock\nTraceback (most recent call last):\n  File "/opt/miniconda/envs/recon/bin/jupyter-nbconvert", line 8, in <module>\n    sys.exit(main())\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_core/application.py", line 283, in launch_instance\n    super().launch_instance(argv=argv, **kwargs)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/traitlets/config/application.py", line 1075, in launch_instance\n    app.start()\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/nbconvertapp.py", line 420, in start\n    self.convert_notebooks()\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/nbconvertapp.py", line 597, in convert_notebooks\n    self.convert_single_notebook(notebook_filename)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/nbconvertapp.py", line 563, in convert_single_notebook\n    output, resources = self.export_single_notebook(\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/nbconvertapp.py", line 487, in export_single_notebook\n    output, resources = self.exporter.from_filename(\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/exporters/templateexporter.py", line 390, in from_filename\n    return super().from_filename(filename, resources, **kw)  # type:ignore[return-value]\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/exporters/exporter.py", line 201, in from_filename\n    return self.from_file(f, resources=resources, **kw)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/exporters/templateexporter.py", line 396, in from_file\n    return super().from_file(file_stream, resources, **kw)  # type:ignore[return-value]\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/exporters/exporter.py", line 220, in from_file\n    return self.from_notebook_node(\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/exporters/html.py", line 278, in from_notebook_node\n    html, resources = super().from_notebook_node(nb, resources, **kw)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/exporters/templateexporter.py", line 412, in from_notebook_node\n    nb_copy, resources = super().from_notebook_node(nb, resources, **kw)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/exporters/exporter.py", line 154, in from_notebook_node\n    nb_copy, resources = self._preprocess(nb_copy, resources)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/exporters/exporter.py", line 353, in _preprocess\n    nbc, resc = preprocessor(nbc, resc)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/preprocessors/base.py", line 48, in __call__\n    return self.preprocess(nb, resources)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbconvert/preprocessors/execute.py", line 97, in preprocess\n    with self.setup_kernel():\n  File "/opt/miniconda/envs/recon/lib/python3.10/contextlib.py", line 135, in __enter__\n    return next(self.gen)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbclient/client.py", line 600, in setup_kernel\n    self.start_new_kernel(**kwargs)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_core/utils/__init__.py", line 165, in wrapped\n    return loop.run_until_complete(inner)\n  File "/opt/miniconda/envs/recon/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete\n    return future.result()\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/nbclient/client.py", line 550, in async_start_new_kernel\n    await ensure_async(self.km.start_kernel(extra_arguments=self.extra_arguments, **kwargs))\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_core/utils/__init__.py", line 198, in ensure_async\n    result = await obj\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 96, in wrapper\n    raise e\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 87, in wrapper\n    out = await method(self, *args, **kwargs)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 435, in _async_start_kernel\n    kernel_cmd, kw = await self._async_pre_start_kernel(**kw)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 397, in _async_pre_start_kernel\n    self.kernel_spec,\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/manager.py", line 195, in kernel_spec\n    self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)\n  File "/opt/miniconda/envs/recon/lib/python3.10/site-packages/jupyter_client/kernelspec.py", line 285, in get_kernel_spec\n    raise NoSuchKernel(kernel_name)\njupyter_client.kernelspec.NoSuchKernel: No such kernel named slidelock\n\nERROR conda.cli.main_run:execute(125): `conda run jupyter nbconvert --execute --to html example.ipynb` failed. (See above for error)\n```\n\n### **Instructions**\n1. Carefully review **Dockerfile**, output of building docker image and output of running docker image, give your\nthoughts and advice as the following format:\n```\n**Thoughts**: you thoughts here\n``` \n2. Be precise and support your reasoning with evidence from the input.\n\n### **Notes**\n- We are generating Dockerfile over multiple rounds, your thoughts and the output of this step will be persisted, \nwe\'ll continue with the next round accordingly\n'"""
    )
    state:DockerGenerationWorkflowState = step.execute(state)
    assert state is not None
    assert state["plan_actions"] is not None
    assert state["plan_thoughts"] is not None


