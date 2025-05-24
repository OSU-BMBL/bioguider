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

@pytest.mark.skip()
def test_DockerGenerationPlanStep(llm, step_callback):
    thoughts = 'The `Docker build` process failed because the package `mappy` attempts to compile and lacks the required GCC compiler. To resolve this, modify the Dockerfile to include installation commands for essential build tools like GCC and make before setting up the Conda environment. Additionally, consider verifying the `environment.yml` file for Conda compatibility and addressing deprecation warnings for future stability.'
    error_msg = 'error:\n#12 187.7   DEPRECATION: Building \'mappy\' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of \'mappy\'. Discussion can be found at https://github.com/pypa/pip/issues/6334\n#12 187.7   error: subprocess-exited-with-error\n#12 187.7   \n#12 187.7   × python setup.py bdist_wheel did not run successfully.\n#12 187.7   │ exit code: 1\n#12 187.7   ╰─> [33 lines of output]\n#12 187.7       /opt/conda/envs/recon/lib/python3.10/site-packages/setuptools/__init__.py:94: _DeprecatedInstaller: setuptools.installer and fetch_build_eggs are deprecated.\n#12 187.7       !!\n#12 187.7       \n#12 187.7               ********************************************************************************\n#12 187.7               Requirements should be satisfied by a PEP 517 installer.\n#12 187.7               If you are using pip, you can try `pip install --use-pep517`.\n#12 187.7               ********************************************************************************\n#12 187.7       \n#12 187.7       !!\n#12 187.7         dist.fetch_build_eggs(dist.setup_requires)\n#12 187.7       /opt/conda/envs/recon/lib/python3.10/site-packages/setuptools/dist.py:759: SetuptoolsDeprecationWarning: License classifiers are deprecated.\n#12 187.7       !!\n#12 187.7       \n#12 187.7               ********************************************************************************\n#12 187.7               Please consider removing the following classifiers in favor of a SPDX license expression:\n#12 187.7       \n#12 187.7               License :: OSI Approved :: MIT License\n#12 187.7       \n#12 187.7               See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.\n#12 187.7               ********************************************************************************\n#12 187.7       \n#12 187.7       !!\n#12 187.7         self._finalize_license_expression()\n#12 187.7       running bdist_wheel\n#12 187.7       running build\n#12 187.7       running build_ext\n#12 187.7       Compiling python/mappy.pyx because it changed.\n#12 187.7       [1/1] Cythonizing python/mappy.pyx\n#12 187.7       building \'mappy\' extension\n#12 187.7       creating build/temp.linux-x86_64-cpython-310\n#12 187.7       creating build/temp.linux-x86_64-cpython-310/python\n#12 187.7       gcc -pthread -B /opt/conda/envs/recon/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/conda/envs/recon/include -fPIC -O2 -isystem /opt/conda/envs/recon/include -fPIC -I. -I/opt/conda/envs/recon/include/python3.10 -c align.c -o build/temp.linux-x86_64-cpython-310/align.o -DHAVE_KALLOC -msse4.1\n#12 187.7       error: command \'gcc\' failed: No such file or directory\n#12 187.7       [end of output]\n#12 187.7   \n#12 187.7   note: This error originates from a subprocess, and is likely not a problem with pip.\n#12 187.7   ERROR: Failed building wheel for mappy\n#12 187.7 ERROR: Failed to build installable wheels for some pyproject.toml based projects (mappy)\n#12 187.7 \n#12 187.8 Ran pip subprocess with arguments:\n#12 187.8 [\'/opt/conda/envs/recon/bin/python\', \'-m\', \'pip\', \'install\', \'-U\', \'-r\', \'/workspace/condaenv.jo4uj44l.requirements.txt\', \'--exists-action=b\']\n#12 187.8 Pip subprocess output:\n#12 187.8 Collecting mappy==2.24 (from -r /workspace/condaenv.jo4uj44l.requirements.txt (line 1))\n#12 187.8   Downloading mappy-2.24.tar.gz (140 kB)\n#12 187.8   Preparing metadata (setup.py): started\n#12 187.8   Preparing metadata (setup.py): finished with status \'done\'\n#12 187.8 Collecting seaborn==0.12.2 (from -r /workspace/condaenv.jo4uj44l.requirements.txt (line 2))\n#12 187.8   Downloading seaborn-0.12.2-py3-none-any.whl.metadata (5.4 kB)\n#12 187.8 Requirement already satisfied: numpy!=1.24.0,>=1.17 in /opt/conda/envs/recon/lib/python3.10/site-packages (from seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (1.23.2)\n#12 187.8 Requirement already satisfied: pandas>=0.25 in /opt/conda/envs/recon/lib/python3.10/site-packages (from seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (1.4.3)\n#12 187.8 Requirement already satisfied: matplotlib!=3.6.1,>=3.1 in /opt/conda/envs/recon/lib/python3.10/site-packages (from seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (3.4.3)\n#12 187.8 Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/recon/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (0.12.1)\n#12 187.8 Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/recon/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (1.4.7)\n#12 187.8 Requirement already satisfied: pillow>=6.2.0 in /opt/conda/envs/recon/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (11.2.1)\n#12 187.8 Requirement already satisfied: pyparsing>=2.2.1 in /opt/conda/envs/recon/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (3.2.3)\n#12 187.8 Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/recon/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (2.9.0.post0)\n#12 187.8 Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/recon/lib/python3.10/site-packages (from pandas>=0.25->seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (2025.2)\n#12 187.8 Requirement already satisfied: six>=1.5 in /opt/conda/envs/recon/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /workspace/condaenv.jo4uj44l.requirements.txt (line 2)) (1.17.0)\n#12 187.8 Downloading seaborn-0.12.2-py3-none-any.whl (293 kB)\n#12 187.8 Building wheels for collected packages: mappy\n#12 187.8   Building wheel for mappy (setup.py): started\n#12 187.8   Building wheel for mappy (setup.py): finished with status \'error\'\n#12 187.8   Running setup.py clean for mappy\n#12 187.8 Failed to build mappy\n#12 187.8 \n#12 187.8 \x08\x08failed\n#12 187.8 \n#12 187.8 CondaEnvException: Pip failed\n#12 187.8 \n#12 ERROR: process "/bin/sh -c conda env create -f environment.yml && conda clean -a" did not complete successfully: exit code: 1\n------\n > [7/7] RUN conda env create -f environment.yml && conda clean -a:\n187.8 Building wheels for collected packages: mappy\n187.8   Building wheel for mappy (setup.py): started\n187.8   Building wheel for mappy (setup.py): finished with status \'error\'\n187.8   Running setup.py clean for mappy\n187.8 Failed to build mappy\n187.8 \n187.8 \x08\x08failed\n187.8 \n187.8 CondaEnvException: Pip failed\n187.8 \n------\ndemo-bioguider-92w3j152nl.Dockerfile:14\n--------------------\n  12 |     \n  13 |     # Install the Conda environment\n  14 | >>> RUN conda env create -f environment.yml && conda clean -a\n  15 |     \n  16 |     # Activate Conda environment and set it as the default\n--------------------\nERROR: failed to solve: process "/bin/sh -c conda env create -f environment.yml && conda clean -a" did not complete successfully: exit code: 1\n'
    dockerfile_content = '# Dockerfile\nFROM continuumio/miniconda3:latest\n\n# Set working directory inside the container\nWORKDIR /workspace\n\n# Copy repository files into the container\nCOPY environment.yml environment.yml\nCOPY demo/example.py demo/example.py\nCOPY demo/coordinate.pkl demo/coordinate.pkl\nCOPY demo/rgb.pkl demo/rgb.pkl\n\n# Install the Conda environment\nRUN conda env create -f environment.yml && conda clean -a\n\n# Activate Conda environment and set it as the default\nENV PATH /opt/conda/envs/recon/bin:$PATH\nENV CONDA_DEFAULT_ENV recon\n\n# Command to run the demo example\nCMD ["python", "demo/example.py"]\n'
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
        step_dockerfile_content=dockerfile_content,
        step_thoughts=thoughts,
        step_output=error_msg,
    )
    state:DockerGenerationWorkflowState = step.execute(state)
    assert state is not None
    assert state["plan_actions"] is not None
    assert state["plan_thoughts"] is not None

def test_DockerGenerationPlanStep_1(llm, step_callback):
    thoughts = """The error encountered during the image build is due to a failure in constructing the mappy package, likely because of unmet system-level dependencies or compatibility issues. It is advised to check the actual requirements for mappy, ensure their installation prior to the conda environment setup, and verify the compatibility of packages in the 'environment.yml'. Systematic debugging of dependency installation and environmental setups can resolve this issue."""
    error_msg = 'irement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/recon/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /app/condaenv.uh5vaa2i.requirements.txt (line 2)) (2.9.0.post0)\n#11 149.4 Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/recon/lib/python3.10/site-packages (from pandas>=0.25->seaborn==0.12.2->-r /app/condaenv.uh5vaa2i.requirements.txt (line 2)) (2025.2)\n#11 149.4 Requirement already satisfied: six>=1.5 in /opt/conda/envs/recon/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn==0.12.2->-r /app/condaenv.uh5vaa2i.requirements.txt (line 2)) (1.17.0)\n#11 149.4 Downloading seaborn-0.12.2-py3-none-any.whl (293 kB)\n#11 149.4 Building wheels for collected packages: mappy\n#11 149.4   Building wheel for mappy (setup.py): started\n#11 149.4   Building wheel for mappy (setup.py): finished with status \'error\'\n#11 149.4   Running setup.py clean for mappy\n#11 149.4 Failed to build mappy\n#11 149.4 \n#11 149.4 \x08\x08failed\n#11 149.4 \n#11 149.4 CondaEnvException: Pip failed\n#11 149.4 \n#11 ERROR: process "/bin/sh -c conda env create -f environment.yml && conda clean -afy" did not complete successfully: exit code: 1\n------\n > [6/6] RUN conda env create -f environment.yml && conda clean -afy:\n149.4 Building wheels for collected packages: mappy\n149.4   Building wheel for mappy (setup.py): started\n149.4   Building wheel for mappy (setup.py): finished with status \'error\'\n149.4   Running setup.py clean for mappy\n149.4 Failed to build mappy\n149.4 \n149.4 \x08\x08failed\n149.4 \n149.4 CondaEnvException: Pip failed\n149.4 \n------\ndemo-bioguider-519e9t9bdx.Dockerfile:32\n--------------------\n  30 |     \n  31 |     # Create and activate conda environment\n  32 | >>> RUN conda env create -f environment.yml && conda clean -afy\n  33 |     \n  34 |     # Activate the environment for the container\n--------------------\nERROR: failed to solve: process "/bin/sh -c conda env create -f environment.yml && conda clean -afy" did not complete successfully: exit code: 1\n'
    dockerfile_content = '# Dockerfile\n\n# Start from a lightweight Ubuntu base image\nFROM ubuntu:20.04\n\n# Set environment variables to minimize Docker output noise\nENV DEBIAN_FRONTEND=noninteractive\n\n# Install necessary system dependencies\nRUN apt-get update && apt-get install -y \\\n    wget \\\n    build-essential \\\n    && apt-get clean\n\n# Set up Miniconda for Python/Conda environment management\nRUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \\\n    bash miniconda.sh -b -p /opt/conda && \\\n    rm miniconda.sh && \\\n    /opt/conda/bin/conda init && \\\n    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh\n\n# Add conda executables to PATH\nENV PATH="/opt/conda/bin:$PATH"\n\n# Copy repository files into the container\nCOPY . /app\n\n# Set the working directory to the repository root\nWORKDIR /app\n\n# Create and activate conda environment\nRUN conda env create -f environment.yml && conda clean -afy\n\n# Activate the environment for the container\nENV CONDA_DEFAULT_ENV=recon\nENV PATH="/opt/conda/envs/recon/bin:$PATH"\n\n# Set entry command to run the demo example\nCMD ["python", "demo/example.py"]\n'
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
        step_dockerfile_content=dockerfile_content,
        step_thoughts=thoughts,
        step_output=error_msg,
    )
    state:DockerGenerationWorkflowState = step.execute(state)
    assert state is not None
    assert state["plan_actions"] is not None
    assert state["plan_thoughts"] is not None

