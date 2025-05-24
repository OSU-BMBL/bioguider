
import pytest
import os
from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate

from bioguider.agents.dockergeneration_task_utils import (
    extract_python_file_from_notebook_tool,
    prepare_provided_files_string,
    write_file_tool,
    generate_Dockerfile_tool,
)
from bioguider.agents.agent_utils import (
    read_directory,
    read_file,
    generate_repo_structure_prompt,
)
from bioguider.agents.dockergeneration_execute_step import (
    DockerGenerationExecuteStep,
    DockerGenerationWorkflowState,
)
from bioguider.agents.python_ast_repl_tool import CustomPythonAstREPLTool
from bioguider.utils.file_utils import extract_code_from_notebook

PLAN_THOUGHTS = """### Reasoning Process\n\nThe current task is to prepare a working Dockerfile to run the demo example from the given repository. The intermediate Dockerfile failed due to a missing kernel (`slidelock`). To resolve this issue and ensure full reproducibility, the following steps are necessary:\n\n**Step 1**: Identify the root cause of the missing kernel issue.  \nThe error indicates that the kernel `slidelock` is unavailable during execution. This kernel name is specific to the environment of the author of the repository. We need to either install the kernel explicitly or decouple the code from this kernel dependency by running the Python script directly.\n\n**Step 2**: Extract notebook code into a Python script for easier execution.  \nPython notebooks are by default executed using a Jupyter environment, and they rely on kernel specifications. To avoid errors similar to the missing kernel issue (`slidelock`), we should extract the code in the notebook to a Python file. This allows us to run the demo independently using the configured environment.\n\n**Step 3**: Update the `environment.yml` file if required.  \nWe should ensure that the environment.yml file has all necessary dependencies installed for the demo example (e.g., UMAP, pandas, matplotlib, seaborn, etc.). These dependencies appear present based on provided data, so no action may be needed here.\n\n**Step 4**: Generate a new Dockerfile.  \nThe Dockerfile will copy the repository files, install the environment, and execute the extracted Python script from the notebook (`example.py`) instead of running the notebook directly.\n\n---\n\n### Final Plan Based on Reasoning\n\nTo address the kernel issue and meet reproducibility requirements:\n1. Extract the code from `demo/example.ipynb` into a Python script (`demo/example.py`) using the `extract_python_file_from_notebook_tool`.\n2. Generate a new Dockerfile using `generate_Dockerfile_tool`, ensuring it installs the required environment using `environment.yml` and executes the Python script (`example.py`).\n\n### Plan\n\nStep: extract_python_file_from_notebook_tool  \nStep Input: {"notebook_path": "demo/example.ipynb", "output_path": "demo/example.py"}\n\nStep: generate_Dockerfile_tool  \nStep Input: {"output_path": "demo-bioguider-5bojhlnyjc.Dockerfile"}
"""

@pytest.mark.skip()
def test_DockerGenerationExecuteStep(llm, step_callback):
    plan_actions = "Step: extract_python_file_from_notebook_tool\nStep Input: {'notebook_path': 'demo/example.ipynb', 'output_path': 'demo/example.py'}\nStep: generate_Dockerfile_tool\nStep Input: {'output_path': 'demo-bioguider-5bojhlnyjc.Dockerfile'}\n"
    plan_thoughts = PLAN_THOUGHTS
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
            func=generate_tool.run,
            description=generate_tool.__class__.__doc__,
            name=generate_tool.__class__.__name__,
        ),
        python_tool,
        StructuredTool.from_function(
            extract_tool.run,
            description=extract_tool.__class__.__doc__,
            name=extract_tool.__class__.__name__,
        )
    ]

    step = DockerGenerationExecuteStep(
        llm=llm,
        repo_path=repo_path,
        repo_structure=repo_structure,
        gitignore_path=gitignore_path,
        custom_tools=custom_tools,
    )
    step.set_generate_Dockerfile_tool(generate_tool)
    state = DockerGenerationWorkflowState(
        llm=llm,
        step_output_callback=step_callback,
        provided_files=provided_files,
        plan_actions=plan_actions,
        plan_thoughts=plan_thoughts,
    )
    state = step.execute(state)
    assert state is not None

@pytest.mark.skip()
def test_DockerGenerationExecuteStep_1(llm, step_callback):
    plan_actions = "Step: extract_python_file_from_notebook_tool\nStep Input: {'notebook_path': 'demo/example.ipynb', 'output_path': 'demo/example.py'}\nStep: generate_Dockerfile_tool\nStep Input: demo-bioguider-8e73uhedis.Dockerfile\n"
    plan_thoughts = '### Reasoning Process:\n\n1. **Understand the Problem**:\n   - The request is to generate a Dockerfile that can prepare the environment and run a simple get-started example based on provided repository files.\n   - The repository contains several Python scripts, an environment specification (`environment.yml`), and a simple demo implementation in the `/demo` directory.\n\n2. **Assess Repository Files**:\n   - The primary focus should be on the `environment.yml`, `/demo/example.ipynb`, and equivalent files (`example.py`) to determine dependencies and how the environment should be prepared.\n   - From the intermediate Dockerfile and its error message, the failure is caused due to an issue with installing the `mappy` package in the conda environment.\n   - The demo makes heavy use of color processing, UMAP embedding, numpy, pandas, seaborn, etc., as evident from `example.ipynb`.\n\n3. **Steps to Solve the Problem**:\n   - **Step 1**: Check if we need to extract the notebook (`example.ipynb`) into a Python file. Doing so avoids requiring Jupyter in the generated Dockerfile and ensures compatibility for running the demo directly.\n   - **Step 2**: Fix the `mappy` dependency installation issues by explicitly specifying prerequisites in the Dockerfile. Although the intermediate error suggests it may have compatibility problems, additional debugging or requirement tracing may resolve this problem.\n   - **Step 3**: Use all dependencies defined in `environment.yml`, ensuring proper installation.\n   - **Step 4**: Structure the Dockerfile such that the environment is prepared, and the demo runs successfully when the container is executed.\n\n4. **Methodology for Tools Usage**:\n   - Use `extract_python_file_from_notebook_tool` for extracting the notebook code.\n   - Write/modify script files with `write_file_tool`.\n   - Generate a final Dockerfile using `generate_Dockerfile_tool`.\n\n5. **Plan Execution**:\n   - Complete all necessary updates for notebooks and scripts using the tools.\n   - Save and create the Dockerfile using `generate_Dockerfile_tool`.\n\n---\n\n### Plan:\n\nStep: extract_python_file_from_notebook_tool  \nStep Input: `demo/example.ipynb`, `demo/example.py`\n\nStep: generate_Dockerfile_tool  \nStep Input: `demo-bioguider-8e73uhedis.Dockerfile`  '
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
            func=generate_tool.run,
            description=generate_tool.__class__.__doc__,
            name=generate_tool.__class__.__name__,
        ),
        python_tool,
        StructuredTool.from_function(
            extract_tool.run,
            description=extract_tool.__class__.__doc__,
            name=extract_tool.__class__.__name__,
        )
    ]

    step = DockerGenerationExecuteStep(
        llm=llm,
        repo_path=repo_path,
        repo_structure=repo_structure,
        gitignore_path=gitignore_path,
        custom_tools=custom_tools,
    )
    step.set_generate_Dockerfile_tool(generate_tool)
    state = DockerGenerationWorkflowState(
        llm=llm,
        step_output_callback=step_callback,
        provided_files=provided_files,
        plan_actions=plan_actions,
        plan_thoughts=plan_thoughts,
    )
    state = step.execute(state)
    assert state is not None

def test_DockerGenerationExecuteStep_2(llm, step_callback):
    plan_actions = "Step: extract_python_file_from_notebook_tool\nStep Input: {'notebook_path': 'demo/example.ipynb', 'output_path': 'demo/example.py'}\nStep: generate_Dockerfile_tool\nStep Input: {'output_path': 'demo-bioguider-e5qjr2p3b6.Dockerfile'}\n"
    plan_thoughts = '### Reasoning Process:\n\n#### **Analysis of Intermediate Error**\nThe error while building the Dockerfile indicates that constructing the `mappy` package via `pip` in the Conda environment failed. This is most likely caused by unmet system-level dependencies, compatibility issues, or the absence of required build tools. Specifically, package `mappy` might require libraries (like C/C++ compilers or specific system packages) to build successfully, and the current system environment may not provide them.\n\n#### **Resolution Strategy**\n1. **System Dependency Installation**: We need to ensure that all system-level dependencies required to build the `mappy` package are installed before creating the Conda environment. This might include development tools such as `gcc`, `g++`, and other libraries.\n2. **Environment Validation**: Confirm that the dependencies listed in the `environment.yml` file are compatible and resolve any conflicts. This specifically includes testing the installation across different Python versions if necessary.\n3. **Demo Extraction**: Instead of running the demo `example.ipynb` file directly, convert it into a Python script (`demo/example.py`) and ensure that it is executed in Docker without requiring Jupyter Notebook.\n4. **Container Debugging**: The Dockerfile should be debug-friendly to handle further build issues.\n\n#### **Solution Plan**\nHere is the plan to resolve the issue:\n1. Install additional system-level dependencies required to build problematic packages like `mappy`.\n2. Convert the Jupyter Notebook `example.ipynb` to a Python script (`example.py`) for execution in the demo.\n3. Replace the problematic Conda environment creation with a more robust installation process:\n   - Install packages that can be installed directly.\n   - Use `pip` separately for specific packages like `mappy`.\n4. Test the adjusted Dockerfile by generating it again.\n\n#### **Execution Plan**\nThe next steps involve:\n1. Using `extract_python_file_from_notebook_tool` to extract Python code from `demo/example.ipynb` and save it to a script file (`demo/example.py`).\n2. Writing a new Dockerfile to include system dependencies, handle environment creation errors, and improve resiliency.\n3. Use the `generate_Dockerfile_tool` tool to generate the updated Dockerfile.\n\n---\n\n### Plan for Execution\n\nStep: extract_python_file_from_notebook_tool  \nStep Input: `demo/example.ipynb`, `demo/example.py`\n\nStep: generate_Dockerfile_tool  \nStep Input: `demo-bioguider-e5qjr2p3b6.Dockerfile`'
    step_error='irement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/recon/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /app/condaenv.uh5vaa2i.requirements.txt (line 2)) (2.9.0.post0)\n#11 149.4 Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/recon/lib/python3.10/site-packages (from pandas>=0.25->seaborn==0.12.2->-r /app/condaenv.uh5vaa2i.requirements.txt (line 2)) (2025.2)\n#11 149.4 Requirement already satisfied: six>=1.5 in /opt/conda/envs/recon/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /app/condaenv.uh5vaa2i.requirements.txt (line 2)) (1.17.0)\n#11 149.4 Downloading seaborn-0.12.2-py3-none-any.whl (293 kB)\n#11 149.4 Building wheels for collected packages: mappy\n#11 149.4   Building wheel for mappy (setup.py): started\n#11 149.4   Building wheel for mappy (setup.py): finished with status \'error\'\n#11 149.4   Running setup.py clean for mappy\n#11 149.4 Failed to build mappy\n#11 149.4 \n#11 149.4 \x08\x08failed\n#11 149.4 \n#11 149.4 CondaEnvException: Pip failed\n#11 149.4 \n#11 ERROR: process "/bin/sh -c conda env create -f environment.yml && conda clean -afy" did not complete successfully: exit code: 1\n------\n > [6/6] RUN conda env create -f environment.yml && conda clean -afy:\n149.4 Building wheels for collected packages: mappy\n149.4   Building wheel for mappy (setup.py): started\n149.4   Building wheel for mappy (setup.py): finished with status \'error\'\n149.4   Running setup.py clean for mappy\n149.4 Failed to build mappy\n149.4 \n149.4 \x08\x08failed\n149.4 \n149.4 CondaEnvException: Pip failed\n149.4 \n------\ndemo-bioguider-519e9t9bdx.Dockerfile:32\n--------------------\n  30 |     \n  31 |     # Create and activate conda environment\n  32 | >>> RUN conda env create -f environment.yml && conda clean -afy\n  33 |     \n  34 |     # Activate the environment for the container\n--------------------\nERROR: failed to solve: process "/bin/sh -c conda env create -f environment.yml && conda clean -afy" did not complete successfully: exit code: 1\n'
    step_dockerfile_content='```dockerfile\n# Dockerfile\n\n# Start from a lightweight Ubuntu base image\nFROM ubuntu:20.04\n\n# Set environment variables to minimize Docker output noise\nENV DEBIAN_FRONTEND=noninteractive\n\n# Install necessary system dependencies for Python/C++ development\nRUN apt-get update && apt-get install -y \\\n    wget \\\n    build-essential \\\n    gcc \\\n    g++ \\\n    make \\\n    && apt-get clean\n\n# Set up Miniconda for Python/Conda environment management\nRUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \\\n    bash miniconda.sh -b -p /opt/conda && \\\n    rm miniconda.sh && \\\n    /opt/conda/bin/conda init && \\\n    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh\n\n# Add conda executables to PATH\nENV PATH="/opt/conda/bin:$PATH"\n\n# Copy repository files into the container\nCOPY . /app\n\n# Set the working directory to the repository root\nWORKDIR /app\n\n# Install Conda dependencies from environment.yml\nRUN conda env create -f environment.yml && conda clean -afy\n\n# Activate the environment for the container\nENV CONDA_DEFAULT_ENV=recon\nENV PATH="/opt/conda/envs/recon/bin:$PATH"\n\n# Install remaining dependencies with pip (e.g., mappy)\nRUN pip install mappy==2.24 seaborn==0.12.2\n\n# Set entry command to run the demo example\nCMD ["python", "demo/example.py"]\n```'
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
            func=generate_tool.run,
            description=generate_tool.__class__.__doc__,
            name=generate_tool.__class__.__name__,
        ),
        python_tool,
        StructuredTool.from_function(
            extract_tool.run,
            description=extract_tool.__class__.__doc__,
            name=extract_tool.__class__.__name__,
        )
    ]

    step = DockerGenerationExecuteStep(
        llm=llm,
        repo_path=repo_path,
        repo_structure=repo_structure,
        gitignore_path=gitignore_path,
        custom_tools=custom_tools,
    )
    step.set_generate_Dockerfile_tool(generate_tool)
    state = DockerGenerationWorkflowState(
        llm=llm,
        step_output_callback=step_callback,
        provided_files=provided_files,
        plan_actions=plan_actions,
        plan_thoughts=plan_thoughts,
        step_output=step_error,
        step_dockerfile_content=step_dockerfile_content,
    )
    state = step.execute(state)
    assert state is not None

