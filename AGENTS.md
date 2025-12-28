
# AGENTS.md

## Project Overview

BioGuider implements a modular, multi-agent architecture for systematically evaluating the quality, completeness, and correctness of documentation in open-source biomedical software repositories.

The overall evaluation workflow is decomposed into two high-level modules:

1. **Collect Module** — identifies, retrieves, and structures documentation-relevant artifacts from the repository
2. **Evaluation Module** — assesses the collected artifacts using predefined quality and consistency criteria

Each module is implemented as a coordinated set of large language model (LLM) agents equipped with task-specific tools.

---

## Collect Module

### Purpose

The **Collect Module** is responsible for discovering, extracting, and organizing all repository files that are relevant to documentation evaluation.
Its goal is to construct a **curated documentation corpus** from repositories with heterogeneous layouts.

Target documentation categories include:

* Project overview / README
* Installation instructions
* User guides and API references
* Tutorials and vignettes

---

### Available Tools

LLM agents in the Collect Module may use the following tools:

* **Directory Reader**
  Traverses the repository directory tree and enumerates files and subdirectories.

* **File Reader**
  Loads the full content of text-based files, including Markdown, R Markdown, Python, and R source files.

* **Relevance Classifier**
  Determines whether a file is relevant to a specific documentation category
  (`README`, `Installation`, `User Guide / API`, `Tutorial / Vignette`).

* **Content Summarizer**
  Produces concise semantic summaries of documentation files to support downstream reasoning.

* **Content Extractor**
  Extracts specific sections or segments (e.g., installation steps, tutorial workflows, API usage examples) from larger documents.

* **Python AST REPL Tool**
  Executes Python code in a controlled environment to perform repository-level analyses, such as:

  * Counting source files by language
  * Inspecting abstract syntax trees (ASTs)
  * Computing simple repository statistics

---

### Agent Roles

The Collect Module operates using a **plan–execute–verify loop** with three specialized agents.

#### 1. Design Agent

**Responsibility:** High-level planning.

Given the collection objective, the Design Agent generates a structured sequence of actions that may combine reasoning steps and tool invocations.

Example planning actions include:

* Running Python code to quantify repository composition (e.g., number of Python vs. R files)
* Traversing documentation-related directories (e.g., `./man`, `./vignettes`, `./docs`)
* Identifying and summarizing README, `.Rd`, or Markdown files
* Extracting executable tutorial sections from notebooks or vignette files

The output of the Design Agent is an **ordered action plan**.

---

#### 2. Execute Agent

**Responsibility:** Action execution.

The Execute Agent carries out the action plan produced by the Design Agent by invoking the specified tools in sequence. It records all intermediate outputs, including:

* Retrieved file contents
* Extracted documentation sections
* Generated summaries
* Results of Python code execution

These outputs are passed to the Observe Agent for verification.

---

#### 3. Observe Agent

**Responsibility:** Goal verification and feedback.

The Observe Agent inspects the outputs produced by the Execute Agent and determines whether the collection objective has been satisfied.

Specifically, it checks whether:

* All required documentation categories are present
* Extracted content is sufficiently complete and unambiguous
* Additional files or sections need to be collected

If gaps or deficiencies are detected, the Observe Agent generates corrective feedback or refinement suggestions, which may trigger an additional planning cycle.

---

## Evaluation Module

### Purpose

The **Evaluation Module** assesses the collected documentation artifacts against predefined quality criteria, including:

* Completeness
* Clarity and readability
* Reproducibility
* Technical correctness

---

### Documentation Quality Assessment

For each documentation category (`README`, `Installation`, `User Guide / API`, `Tutorial / Vignette`), the Evaluation Agent applies category-specific evaluation rubrics.

These rubrics assess aspects such as:

* Discoverability and organization
* Dependency specification
* Workflow clarity
* Example completeness
* Result interpretability

---

### Code–Documentation Consistency Evaluation

For **User Guides / APIs** and **Tutorials / Vignettes**, BioGuider additionally evaluates consistency between documentation and source code.

This process consists of two stages.

#### 1. Source Code Structure Indexing

BioGuider scans the repository source code and constructs a structured code index capturing:

* Function and class names
* Argument names and signatures
* Return values
* Inline documentation and docstrings

---

#### 2. Consistency Verification

The Evaluation Agent compares documented code usage against the indexed source code structure. It verifies:

* Whether referenced functions and classes exist
* Whether argument names, order, and defaults are correct
* Whether documented behavior matches source-level definitions

This analysis enables BioGuider to identify:

* Outdated examples
* Incorrect API usage
* Documentation–implementation mismatches

Such issues can negatively impact usability and reproducibility.

---

## Summary

By integrating **planning, execution, observation, and evaluation** within a modular multi-agent architecture, BioGuider provides a scalable and automated framework for documentation assessment.

The separation between the Collect and Evaluation modules ensures robustness across diverse repository structures while enabling fine-grained, code-aware analysis of documentation quality and consistency.

---

