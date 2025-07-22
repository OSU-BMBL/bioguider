from enum import Enum
from langchain_core.prompts import ChatPromptTemplate

USER_INSTRUCTION = """Do not give the final result immediately. First, explain your reasoning process step by step, then provide the answer."""

EVALUATION_ITEMS = [
    ("1. Clarity & Readability", 20),
    ("2. Completeness", 20), 
    ("3. Organization & Navigation", 10),
    ("4. Examples & Tutorials", 10), 
    ("5. Maintainability & Updates", 15), 
    ("6. Accessibility & Formatting", 15), 
]

EVALUATION_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""Please act as both a **biomedical researcher** and an **experienced software developer** to evaluate the documentation quality of a GitHub repository using the evaluation criteria below.

### **Evaluation Criteria (Total: 100 points)**

1. **Clarity & Readability (20 points)** - Is the documentation written in a clear, concise, and easy-to-understand manner?
2. **Completeness (20 points)** - Does the documentation cover all essential information needed for understanding, usage, and further development?
3. **Organization & Navigation (10 points)** - Is the structure logical and easy to navigate? Are key sections easy to find?
4. **Examples & Tutorials (10 points)** - Are there sufficient examples or tutorials to help users get started and understand core functionality?
5. **Maintainability & Updates (15 points)** - Does the documentation reflect ongoing maintenance and version history (e.g., changelogs, version tags)?
6. **Accessibility & Formatting (15 points)** - Is the documentation well-formatted and easy to read (e.g., Markdown formatting, appropriate use of code blocks, headers, etc.)?
### **Repository Structure Overview**  
_(f = file, d = directory)_
```
{repository_structure}
```""")

EVALUATION_ITEM_PROMPT = ChatPromptTemplate.from_template("""Here are the content of files or directories in the repository that you need to take into account:
{files_or_directories}

### **Instructions**

Let's begin by evaluating **Criterion {evaluation_item}*.

- If the information provided is **sufficient**, please proceed with your evaluation using the following format:
```
{evaluation_item} ({score_point} points)  
    a. Score: [score out of {score_point}]  
    b. Reason: [brief explanation justifying the score]  
```
- If the information provided is **insufficient**, do **not** attempt to evaluate. Instead, list the specific files or directories for which you need more detail, using the format below:
```
[files/directories needed for evaluation]
```""")


## goal: identify project type
IDENTIFICATION_GOAL_PROJECT_TYPE = """Identify the following key attribute of the repository:
  **project type**: The primary functional type of the project.  
    Options and their definitions:  
    - **package**: A reusable Python or R library intended to be imported by other software.  
    - **application**: A standalone Python or R program that can be directly executed by users.  
    - **pipeline**: A biomedical data processing workflow that integrates multiple tools or steps.
    - **unknown type**: Use this only if the type cannot be determined reliably from available information.
  **Notes**:
    1. The project can be identified as one of the above project type.
    2. The project may server as multiple project types, like package & pipeline, standalone application & package,
      However, you need to investigate closely to find out the primary project type.
    3. Do **not** rely heavily on directories like 'benchmark/' or 'tests/' when determining the project type, as they are often auxiliary."""

## goal: identify primary language
IDENTIFICATION_GOAL_PRIMARY_LANGUAGE = """Identify the following key attribute of the repository:
  **primary language**: The primary language of the project.  
    Options and their definitions:  
    - **python**: Python language
    - **R**: R language
    - **unknown type**: Use this only if the type cannot be determined reliably from available information.
  **Notes**:
    The project can be identified as one of the above primary language."""

## goal: identify meta data: repo name, owner, description, license
IDENTIFICATION_GOAL_META_DATA = """Identify the following meta data of the repository:
  **name**: The repository name.
  **owner**: The repository user or orgnization.
  **description**: The description of the repository.
  **license**: The license of the repository, like 'MIT', 'Apache 2.0' or 'unknown'.

**Notes**: If the above meta data can't be identified, please return 'unknown' or 'N/A'.
"""

COT_USER_INSTRUCTION = "Do not give the answer immediately. First, explain your reasoning process step by step, then provide the answer."
EVALUATION_INSTRUCTION="Before arriving at the conclusion, clearly explain your reasoning step by step. Now, let's begin the evaluation."

class CollectionGoalItemEnum(Enum):
    UserGuide = "User Guide"
    Tutorial = "Tutorials & Vignettes"
    DockerGeneration = "Docker Generation"
    Installation = "Installation"
    License = "License"
    Contributing = "Contributing"
    SoftwarePackageContent = "SoftwarePackageContent"



COLLECTION_GOAL = """Your goal is to collect the names of all files that are relevant to **{goal_item}**.  
**Note:**
 - You only need to collect the **file names**, not their contents."""

COLLECTION_PROMPTS = {
    "UserGuide": {
        "goal_item": "User Guide",
        "related_file_description": """A document qualifies as a **User Guide** if it includes **at least one** of the following elements.
If **any one** of these is present, the document should be classified as a User Guide — full coverage is **not required**:
 - Overview: A brief introduction to the software, its purpose, and its intended audience.
 - Installation Instructions: Step-by-step setup procedures.
 - Input/Output Specifications: Detailed information on the data the software accepts and produces.
 - Configuration Options: Descriptions of settings and parameters that can be adjusted.
 - Function/Interface Listings: Comprehensive lists of available functions or interfaces, including their descriptions, parameters, and return values.
 - Mathematical Equations/Numerical Methods: Embedded documentation explaining the underlying mathematical concepts or algorithms.
 - Developer Guidance: Instructions on how to extend the software or contribute to its development.
 **Do not** classify the document as a User Guide if it primarily serves as a Tutorial or Example. Such documents typically include:
 - Sample Datasets: Example data used to illustrate functionality.
 - Narrative Explanations: Story-like descriptions guiding the user through examples.
 - Code Walkthroughs: Detailed explanations of code snippets in a tutorial format.
**Do not** classify the document as a User Guide if it is souce code or a script (*.py, *.R) that is not intended for end-user interaction.
 - You can include directory names if all files in the directory are relevant to the goal item.""",
    },
    "Tutorial": {
        "goal_item": "Tutorials & Vignettes",
        "related_file_description": """
**Tutorials and Vignettes** are instructional documents or interactive notebooks that provide step-by-step guidance on using a software package or library. They typically include:
 - Code Examples: Practical code snippets demonstrating how to use the software's features and functions.
 - Explanatory Text: Clear explanations accompanying the code examples to help users understand the concepts and techniques being presented.​
 - Visualizations: Graphical representations of data or results to enhance understanding.
 - Interactive Elements: Features that allow users to experiment with the code in real-time, such as Jupyter notebooks or R Markdown files.​
 - Use Cases: Real-world applications or scenarios where the software can be applied effectively.
 - You can include directory names if all files in the directory are relevant to the goal item.
""",
    },
    "DockerGeneration": {
        "goal_item": "Generating a Dockerfile for reproducibility testing",
        
        "related_file_description": """A document qualifies **Dockerfile Generation** related if it includes **at least one** of the following elements.
If **any one** of these is present, the document should be classified as a Dockerfile — full coverage is **not required**:
 - Existing Docker Configuration
   * Files like `Dockerfile`, `docker-compose.yml`, or any Docker-related build scripts.
 - Installation & Environment Setup
   * Files used to define or install dependencies.
     * Examples: `README.md` `requirements.txt`, `environment.yml`, `setup.py`, `install.R`, `DESCRIPTION`, `pyproject.toml`, etc.
 - Build/Runtime Scripts
   * Shell or batch scripts used for setup, building, or launching the application.
     * Examples: `install.sh`, `build.sh`, `run.sh`, etc.
 - Minimal Code Examples or Get-Started Files
   * Files that demonstrate a minimal working example of the software (e.g., for testing or reproducing results).
     * Examples: `example.py`, `main.py`, `demo.R`, `notebooks/get_started.ipynb`, etc.
     * These should be runnable with minimal configuration.""",
        
        "plan_important_instructions": """- Only include minimal code examples that demonstrate basic functionality.
If multiple example files are found, select only the simplest and most lightweight one that is sufficient to verify the repository works.
 - Give priority to analyzing files whose names include **"install"** or **"Dockerfile"**, as these are most likely to be useful for generating our Dockerfile
 - The total number of collected files should **not exceed 5**.
 - Make sure to include **only one code example**, selecting the most minimal and representative one.
""",
        "observe_important_instructions": """- Only include minimal code examples that demonstrate basic functionality.
If multiple example files are found, select only the simplest and most lightweight one that is sufficient to verify the repository works.
 - Give priority to analyzing files whose names include **"install"** or **"Dockerfile"**, as these are most likely to be useful for generating our Dockerfile
 - The total number of collected files should **not exceed 5**.
 - Make sure to include **only one code example**, selecting the most minimal and representative one.
""",
    },
    "Installation": {
        "goal_item": "Installation Instructions",
        "related_file_description": """A document qualifies as **Installation Instructions** if it includes **at least one** of the following elements.
If **any one** of these is present, the document should be classified as Installation Instructions — full coverage is **not required**:
 - Step-by-step setup procedures for the software.
 - Prerequisites or dependencies that need to be installed before using the software.
 - Configuration steps required to get the software running.
 - Troubleshooting tips related to installation issues.
 - You can include directory names if all files in the directory are relevant to the goal item.""",
        "plan_important_instructions": """ - Give priority to analyzing README file that contain installation instructions and the files whose names include **"install"** or **"setup"**.
- If multiple files are found, select the most comprehensive one that covers the installation process.
- The total number of collected files should **not exceed 3**.
- Identify and select **no more than three** installation instruction files — choose the most comprehensive and representative ones.
""",
        "observe_important_instructions": """ - Give priority to analyzing README file that contain installation instructions and the files whose names include **"install"** or **"setup"**.
- If multiple files are found, select the most comprehensive one that covers the installation process.
- The total number of collected files should **not exceed 3**.
- Identify and select **no more than three** installation instruction files — choose the most comprehensive and representative ones.
""",
    },
    "License": {
        "goal_item": "License Information",
        "related_file_description": """A document qualifies as **License Information** if it includes **at least one** of the following elements.
If **any one** of these is present, the document should be classified as License Information — full coverage is **not required**:
 - A file named `LICENSE`, `LICENSE.txt`, or similar that explicitly states the software's license.
 - A section in the README or documentation that describes the licensing terms.
 - Any file that contains legal information regarding the use, distribution, or modification of the software.
 - You can include directory names if all files in the directory are relevant to the goal item.""",
    },
    "Contributing": {
        "goal_item": "Contributing Guidelines",
        "related_file_description": """A document qualifies as **Contributing Guidelines** if it includes **at least one** of the following elements.
If **any one** of these is present, the document should be classified as Contributing Guidelines — full coverage is **not required**:
 - A file named `CONTRIBUTING.md`, `CONTRIBUTING.rst`, or similar that provides guidelines for contributing to the project.
 - A section in the README or documentation that outlines how to contribute, report issues, or submit pull requests.
 - Any file that contains instructions for developers on how to contribute to the project, including coding standards, testing procedures, and submission processes.
 - You can include directory names if all files in the directory are relevant to the goal item.""",
    },
    "SoftwarePackageContent": {
        "goal_item": "Software Package Content",
        "related_file_description": """A file qualifies as **Software Package Content** if it meets **at least one** of the following elements.
 - A compiled binary file that may be qualified as a compiled standalone software, please carefully analyze a binary file and its file name to identify if it is a compiled standalone software
 - A source code file, like a file whose extension is `.py`, `.R`, `.ipynb`, `.ts`, or `.js`.
 - An example data which is used to demonstrate usage or for tutorial. Image file should not be considered as example data.
""",
        "plan_important_instructions": """ - A comiled standalone software file is non-textual and appears to be in an executable format (e.g., `.exe`, `.dll`, `.so`, `.bin`, `.elf`).
 - A comiled standalone software file **is not a script or compiled library**, that is, It is not a wrapper script (e.g., shell, Python, Python notebook or Rmd) nor a dynamic/shared library meant for linking.
 - When identifying source code file, prioritize analyzing the file's **extension** and **file name** and try to avoid reading file, using check_file_related_tool or summarizing file content.
 - When identifying example data, prioritize analyzing the file's **extension** (like .dat, .csv, .fastq, and so on) and **file name** (like example_data.txt, example.dat, and so on). If extension/name is ambiguous, use summarizing file content to decide.
 - **Note**: You **only need to detect** whether at least **one** compiled standalone software file, **one** source code file and **one** example data file exist — no need to list all such files.
""",
        "observe_important_instructions": """ - A comiled standalone software file is non-textual and appears to be in an executable format (e.g., `.exe`, `.dll`, `.so`, `.bin`, `.elf`).
 - A comiled standalone software file **is not a script or compiled library**, that is, It is not a wrapper script (e.g., shell, Python, Python notebook or Rmd) nor a dynamic/shared library meant for linking.
 - When identifying source code file, prioritize analyzing the file's **extension** and **file name** and try to avoid reading file, using check_file_related_tool or summarizing file content.
 - When identifying example data, prioritize analyzing the file's **extension** (like .dat, .csv, .fastq, and so on) and **file name** (like example_data.txt, example.dat, and so on). If extension/name is ambiguous, use summarizing file content to decide.
 - **Note**: You **only need to detect** whether at least **one** compiled standalone software file, **one** source code file and **one** example data file exist — no need to list all such files.
 - **Final answer format**: If you believe **all relevant files** have been collected:
   Your final answer **must exactly match** the following format: 
   **FinalAnswer:** {{"final_answer": [<N/A or a compiled filename>, <N/A or a source file name>, <N/A or a example data file name>]}} 
   For each category, return a single file name or `"N/A"` if none found. And the return array must exactly follow this order: [<A comiled standalone software file name>, <A source code file name>, <A example data file name>]
   For example, **FinalAnswer:** {{"final_answer": ["N/A", "app.py", "example.csv"]}} indicates:
   * No compiled standalone software found
   * `app.py` found as source code
   * `example.csv` found as example data
""",
    },
}



