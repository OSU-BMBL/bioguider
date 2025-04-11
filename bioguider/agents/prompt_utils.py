from langchain_core.prompts import ChatPromptTemplate

COLLECTION_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
Please act as both a **biomedical researcher** and an **experienced software developer**, to collect {document} documentation from a repository, 
Here is the file structure (level: {file_structure_level}) of the repository:
{file_structure}
                                                            
Here are functions you can call:
{function_descriptions}
                                                            
Please draft a step-by-step plan to collect all {document} documentation, the planned step format should be like this:
1. step description: step_description
   functions to call: a list of functions to call and functions' arguments, if no function to call, return None
""")

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



