from enum import Enum
from langchain_core.prompts import ChatPromptTemplate

USER_INSTRUCTION = """
Do not give the final result immediately. First, explain your reasoning process step by step, then provide the answer.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
"""

EVALUATION_ITEMS = [
    ("1. Clarity & Readability", 20),
    ("2. Completeness", 20), 
    ("3. Organization & Navigation", 10),
    ("4. Examples & Tutorials", 10), 
    ("5. Maintainability & Updates", 15), 
    ("6. Accessibility & Formatting", 15), 
]

EVALUATION_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
Please act as both a **biomedical researcher** and an **experienced software developer** to evaluate the documentation quality of a GitHub repository using the evaluation criteria below.

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
```
""")

EVALUATION_ITEM_PROMPT = ChatPromptTemplate.from_template("""
---

Here are the content of files or directories in the repository that you need to take into account:
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
```
""")


## goal: identify project type
IDENTIFICATION_GOAL_PROJECT_TYPE = """
Identify the following key attribute of the repository:
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
    3. Do **not** rely heavily on directories like 'benchmark/' or 'tests/' when determining the project type, as they are often auxiliary.
"""

## goal: identify primary language
IDENTIFICATION_GOAL_PRIMARY_LANGUAGE = """
Identify the following key attribute of the repository:
  **primary language**: The primary language of the project.  
    Options and their definitions:  
    - **python**: Python language
    - **R**: R language
    - **unknown type**: Use this only if the type cannot be determined reliably from available information.
  **Notes**:
    The project can be identified as one of the above primary language.
"""

COT_USER_INSTRUCTION = "Do not give the answer immediately. First, explain your reasoning process step by step, then provide the answer."

class CollectionGoalItemEnum(Enum):
    UserGuide = "User Guide"
    Tutorial = "Tutorials & Vignettes"



COLLECTION_GOAL = """
Your goal is to collect the names of all files that are relevant to **{goal_item}**.  
Note: You only need to collect the **file names**, not their contents.
"""

COLLECTION_PROMPTS = {
    "UserGuide": {
        "goal_item": "User Guide",
        "related_file_description": """
A document qualifies as a User Guide if it includes:​
 - Installation Instructions: Step-by-step setup procedures.
 - Input/Output Specifications: Detailed information on the data the software accepts and produces.
 - Configuration Options: Descriptions of settings and parameters that can be adjusted.
 - Function/Interface Listings: Comprehensive lists of available functions or interfaces, including their descriptions, parameters, and return values.
 - Mathematical Equations/Numerical Methods: Embedded documentation explaining the underlying mathematical concepts or algorithms.
 - Developer Guidance: Instructions on how to extend the software or contribute to its development.​

**Do not** classify the document as a User Guide if it primarily serves as a Tutorial or Example. Such documents typically include:​
 - Sample Datasets: Example data used to illustrate functionality.
 - Narrative Explanations: Story-like descriptions guiding the user through examples.
 - Code Walkthroughs: Detailed explanations of code snippets in a tutorial format.​
**Do not** classify the document as a User Guide if it is souce code or a script (*.py, *.R) that is not intended for end-user interaction.​

""",
    },
    "Tutorial": {
        "goal_item": "Tutorials & Vignettes",
        "related_file_description": """
Tutorials and Vignettes are instructional documents or interactive notebooks that provide step-by-step guidance on using a software package or library. They typically include:
 - Code Examples: Practical code snippets demonstrating how to use the software's features and functions.​
 - Explanatory Text: Clear explanations accompanying the code examples to help users understand the concepts and techniques being presented.​
 - Visualizations: Graphical representations of data or results to enhance understanding.​
 - Interactive Elements: Features that allow users to experiment with the code in real-time, such as Jupyter notebooks or R Markdown files.​
 - Use Cases: Real-world applications or scenarios where the software can be applied effectively.​
""",
    },
}

