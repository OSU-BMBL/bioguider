# BioGuider Multi-Agent Evaluation Framework
BioGuider implements a modular, multi-agent system to systematically evaluate the quality, completeness, and correctness of documentation in open-source biomedical software repositories. The evaluation workflow is decomposed into two modules: (i) the Collect Module and (ii) the Evaluation Module. Each module is implemented as a coordinated set of large language model (LLM) agents equipped with domain-specific tools.

## 1. Collect Module
The Collect Module identifies, retrieves, and structures repository files that are relevant to documentation evaluation. Its primary objective is to assemble a curated corpus of documentation artifacts, including README files, installation instructions, user guides/API references, and tutorials or vignettes, from heterogeneous repository layouts.

### 1.1 Tooling
To support flexible and repository-agnostic exploration, BioGuider provides the following tools to LLM agents:
- Directory Reader: traverses the repository structure to enumerate files and subdirectories.
- File Reader: loads the full content of text-based files (e.g., Markdown, R Markdown, Python, R source files).
- Relevance Classifier: determines whether a file is relevant to a target documentation category (README, Installation, User Guide/API, Tutorial/Vignette).
- Content Summarizer: produces concise semantic summaries of documentation files to support downstream reasoning.
- Content Extractor: extracts specific sections (e.g., installation steps, tutorial workflows, API usage examples) from larger documents.
- Python AST REPL Tool: executes Python code in a controlled environment to perform repository-level analyses (e.g., counting source files, inspecting abstract syntax trees, or computing simple statistics).

### 1.2 Agent Roles
The Collect Module consists of three specialized agents operating in a plan-execute-verify loop:
- Design Agent: generates a structured sequence of actions that may combine tool usage and reasoning steps (e.g., traversing ./man or ./vignettes, summarizing README/.Rd files, extracting notebook tutorial sections).
- Execute Agent: carries out the action plan produced by the Design Agent, invoking the specified tools in sequence and recording intermediate outputs (file contents, summaries, execution results).
- Observe Agent: inspects the outputs produced by the Execute Agent and evaluates whether the collection objective has been satisfied. If required artifacts are missing, incomplete, or ambiguous, it produces corrective feedback that can trigger additional planning cycles.

## 2. Evaluation Module
The Evaluation Module assesses collected documentation artifacts against predefined quality criteria (completeness, clarity, reproducibility, and technical correctness). Evaluation is performed per documentation category using category-specific rubrics and structured LLM outputs.

### 2.1 Evaluation Pipeline
For each documentation category (README, Installation, User Guide/API, Tutorial/Vignette), BioGuider applies a consistent evaluation pipeline:
1. File normalization and sanitation. Binary files are skipped; oversized files are excluded; HTML is converted to text; notebooks are reduced to code and markdown; braces are escaped to avoid prompt template collisions.
2. Readability analysis. BioGuider computes readability metrics (Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG) and provides them to the evaluator.
3. Structured evaluation. An LLM returns a schema-constrained evaluation that scores category-specific criteria and provides targeted improvement suggestions.
4. Free-form evaluation. For README and Installation, a second LLM pass expands the structured evaluation into detailed, human-readable feedback with quoted snippets and improvement comments.
5. Score aggregation. Subscores are combined via weighted aggregation to compute an overall category score.

### 2.2 Documentation Quality Rubrics
Each category is evaluated with tailored criteria:
- README: availability, readability, project purpose, hardware/software specifications, dependency clarity, license, and contributor/author information.
- Installation: presence of installation instructions, dependency specification, OS compatibility, hardware requirements, and tutorial completeness.
- User Guide/API: readability, context and purpose, error handling guidance, and coverage of usage examples.
- Tutorial/Vignette: readability, setup and dependencies, reproducibility, structure and navigation, executable code quality, result verification, and performance/resource notes.

### 2.3 Code-Documentation Consistency Evaluation
For User Guides/APIs and Tutorials/Vignettes, BioGuider additionally evaluates consistency between documentation and source code.
1. Source Code Structure Indexing. BioGuider scans repository code to build a structured index capturing function/class names, argument signatures, return values, and inline documentation/docstrings.
2. Consistency Verification. The Evaluation Agent compares documented usage against the index to verify that referenced functions/classes exist, argument names and order are correct, and documented behavior matches source-level definitions.

This consistency analysis identifies outdated examples, incorrect API usage, and documentation-implementation mismatches that may hinder reproducibility or usability.

### 2.4 Representative LLM Prompts
To increase transparency and reproducibility, we include representative prompt blocks used by the Evaluation Module (placeholders indicate runtime content injection).

**General evaluation instruction**
```text
Please also clearly explain your reasoning step by step. Now, let's begin the evaluation.
```

**README structured evaluation (extended excerpt)**
```text
You are an expert in evaluating the quality of README files in software repositories.
Your task is to analyze the provided README file and generate a structured quality assessment based on the following criteria.
If a LICENSE file is present in the repository, its content will also be provided to support your evaluation of license-related criteria.
You must provide the evaluation score in your response.
---
### Evaluation Criteria
1. Available: Is the README accessible and present? Output: Yes or No
2. Readability: Evaluate based on readability metrics AND identify specific errors/issues in the text.
   - You must identify and list ALL errors and anomalies (typos, malformed links, markdown errors, image syntax errors, domain term errors, inconsistencies, formatting issues, and other anomalies).
   - For each error, provide the exact text snippet, error type, suggested correction, and explanation.
3. Project Purpose: Is the project's goal or function clearly stated? Output: Yes or No
4. Hardware and Software Requirements: Are hardware/software specs and compatibility details included?
5. Dependencies: Are all necessary software libraries and dependencies clearly listed?
6. License Information: Is license type clearly indicated?
7. Author/Contributor Info: Are contributor or maintainer details provided?
8. Overall Score: Give an overall quality rating of the README.
---
### Readability Metrics
Flesch Reading Ease: {flesch_reading_ease}
Flesch-Kincaid Grade Level: {flesch_kincaid_grade}
Gunning Fog Index: {gunning_fog_index}
SMOG Index: {smog_index}
---
### README Path
{readme_path}
### README Content
{readme_content}
### LICENSE Path
{license_path}
### LICENSE Summarized Content
{license_summarized_content}
```

**Installation structured evaluation (extended excerpt)**
```text
You are an expert in evaluating the quality of installation information in software repositories.
Your task is to analyze the provided files related to installation and generate a structured quality assessment based on the following criteria.
---
### Evaluation Criteria
1. Installation Available: Is the installation section in document (like README.md or INSTALLATION)?
2. Installation Tutorial: Is the step-by-step installation tutorial provided?
3. Number of required Dependencies Installation: The number of dependencies required to install.
4. Compatible Operating System: Is the compatible operating system described?
5. Hardware Requirements: Are hardware requirements described?
6. Overall Score: Give an overall quality rating of the installation information.
---
### Installation Files Provided
{installation_files_content}
```

**User guide evaluation (extended excerpt)**
```text
You are an expert in evaluating the quality of user guide in software repositories.
Your task is to analyze the provided files related to user guide and generate a structured quality assessment based on the following criteria.
---
1. Readability AND Error Detection:
   - Use Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG.
   - You must scan for and identify ALL error instances (typos, malformed links, markdown/RMarkdown errors, bio term errors, function name errors, inline code formatting errors, and other anomalies).
   - List each occurrence separately; do not group similar errors.
2. Arguments and Clarity: describe arguments and their usage with concrete improvement suggestions.
3. Return Value and Clarity: describe return values and meaning with improvement suggestions.
4. Context and Purpose: describe context and purpose with improvement suggestions.
5. Error Handling: describe error handling with improvement suggestions.
6. Usage Examples: describe usage examples with improvement suggestions.
7. Overall Score: output 0-100.
---
### User Guide Content
{userguide_content}
```

**Tutorial evaluation (extended excerpt)**
```text
You are an expert in evaluating the quality of tutorials in software repositories.
Your task is to analyze the provided tutorial file and generate a structured quality assessment based on the following criteria.
---
1. Readability AND Error Detection:
   - Use Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG.
   - You must scan for and identify ALL error instances (typos, malformed links, markdown/RMarkdown errors, bio term errors, function name errors, inline code formatting errors, and other anomalies).
   - List each occurrence separately; do not group similar errors.
2. Coverage: whether it covers major steps, dependencies, prerequisites, setup, and example usage.
3. Reproducibility: whether it provides a clear description of reproducibility.
4. Structure and Navigation: logical sections, TOC/anchors, time estimates.
5. Executable Code Quality: executable and idiomatic code, no hard-coded paths.
6. Result Verification: expected outputs and acceptance criteria.
7. Performance and Resource Notes: CPU/GPU usage, memory, runtime estimates.
---
### Tutorial File Content
{tutorial_file_content}
```

## Summary
By integrating planning, execution, observation, and evaluation within a multi-agent architecture, BioGuider provides a scalable and automated framework for documentation assessment. The separation between collection and evaluation modules ensures robustness across diverse repository structures while enabling fine-grained, code-aware documentation analysis.
