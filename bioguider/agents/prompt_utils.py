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
    The project can be identified as one of the above project type.
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

## plan system prompt
IDENTIFICATION_PLAN_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
### **Goal**
You are an expert developer in the field of biomedical domain. Your goal is:
{goal}

### **Repository File Structure**
Here is the 3-level file structure of the repository (f - file, d - directory):
{repo_structure}

### **Function Tools**
You are provided the following function tools:
{tools}

### Intermediate Steps
Hers are the intermediate steps results:
{intermediate_steps}

### Intermediate Thoughts
Analysis: {intermediate_analysis}
Thoughts: {intermediate_thoughts}

### **Instruction**
All the results in each round will be persisted, meaning that states and variables will persisted through
multiple rounds of plan execution. Be sure to take advantage of this by developing your collection plan
incrementally and reflect on the intermediate observations at each round, instead of coding up everything 
in one go. Be sure to take only one or two actions in each step.

### **Output**
You plan should follow this format:
Step: tool name, should be one of {tool_names}
Step Input: file name or directory name
Step: tool name, should be one of {tool_names}
Step Input: file name or directory name
""")

## executioin system prompt
IDENTIFICATION_EXECUTION_SYSTEM_PROMPT = """
You are an expert Python developer.

You are given a **plan** and are expected to complete it using Python code and the available tools.

---

### **Available Tools**
{tools}

---

### **Your Task**

Execute the plan step by step using the format below:

```
Thought: Describe what you are thinking or planning to do next.  
Action: The tool you are going to use (must be one of: {tool_names})  
Action Input: The input to the selected action  
Observation: The result returned by the action  
```

You may repeat the **Thought → Action → Action Input → Observation** loop as many times as needed.

Once the plan is fully completed, output the result in the following format:
```
Thought: I have completed the plan.  
Final Answer: Summarize all actions taken in the following format:

**{{Action Input}}**: ({{Action}}) - Action Result: {{Action Output}}
**{{Action Input}}**: ({{Action}}) - Action Result: {{Action Output}}
...
```

---

### **Example**
```
Action: read_file  
Action Input: README.md  
Observation: ### Scanpy \n### Installation ...  
...  
Final Answer:  
**README.md**: (analyze_file_tool) - ### Scanpy \n ### Installation ...
...
```

---

### Notes
Please follow the plan exactly as specified. Do not take any actions outside of the outlined steps.

### **Plan**
{plan_steps}

### **Actions Already Taken**
{agent_scratchpad}

---

{input}
"""

## observation system prompt
IDENTIFICATION_OBSERVATION_SYSTEM_PROMPT = """
Your goal is:
{goal}

### **Repository File Structure**
Here is the 3-level file structure of the repository (f - file, d - directory):
{repo_structure}

### **Intermediate Output**
{intermediate_output}

### **Instructions**
Carefully review the **Goal**, **Repository File Structure**, and **Intermediate Output**.
- If you believe the goal **can be achieved**, proceed as follows:  
  - Provide your reasoning under **Analysis**  
  - Then provide your result under **FinalAnswer**  
  ```
  **Analysis**: your analysis here  
  **FinalAnswer**: your final answer here
  ```
- If the information is **not sufficient** to achieve the goal, simply explain why under **Thoughts**:  
  ```
  **Thoughts**: your thoughts here
  ```
Be precise and support your reasoning with evidence from the input.

### Notes
We are collecting information over multiple rounds, so please **do not rush to provide a Final Answer**.  
If you find the current information insufficient, share your reasoning or thoughts instead—we’ll continue with the next round accordingly.
"""

COT_USER_INSTRUCTION = "Do not give the final result immediately. First, explain your reasoning process step by step, then provide the answer."


