
### Testing module
Purpose: test reproducibility by automatically building and running a Docker image for a minimal "get-started" example.

The testing module is implemented as a plan-execute-observe loop:
- Design Agent (plan step): builds a tool-level plan from the repo structure, provided files, and any intermediate Dockerfile/error messages. The plan can include:
  - `extract_python_file_from_notebook_tool` to extract runnable code from notebooks
  - `write_file_tool` to create or overwrite a minimal demo script
  - `generate_Dockerfile_tool` as the final action to produce `demo-bioguider-<id>.Dockerfile`
- Execute Agent (execute step): follows the plan strictly using only the specified tools and records the generated Dockerfile path and tool outputs.
- Observe Agent (observe step): runs `docker build` and `docker run` on the generated Dockerfile, captures errors, and returns structured observations used to re-plan. If build and run succeed, the Dockerfile content is returned as the final answer.

Key behavior encoded in the pipeline:
- The plan is re-generated when build/run errors occur, using the intermediate Dockerfile and error context.
- Dockerfile generation must be the last planned action to ensure the final artifact reflects the latest fixes.
- Execution does not improvise; it only runs the planned tool actions.

### Report module
The report module converts assessment JSON into a human‑readable, publication‑ready report that is consistent across repositories.

Inputs
- Assessment JSON with per‑file scores, issues, and suggestions.
- Optional testing outputs (e.g., Docker build/run results, error logs).
- Optional metadata (repo URL, commit SHA, model ID, evaluation timestamp).

Processing
- Normalize and validate: parse JSON into a common schema and check for missing fields.
- Aggregate: group findings by documentation category (README, installation, user guide, tutorial) and by file.
- Prioritize: surface “Poor/Fair” items first and highlight high‑impact failures (reproducibility, broken links, missing dependencies).
- Summarize: produce concise narratives with concrete evidence snippets and references.
- Format: generate a Markdown report, then render to HTML/PDF.

Report structure
- Executive summary with overall scores and key risks.
- Category sections with per‑file tables (score, issues, suggested fixes).
- Reproducibility section summarizing testing outcomes and failure modes.
- Appendix with full issue lists, evaluation settings, and links to source files.

Traceability and reproducibility
- Every reported issue links back to the source file and section.
- Report includes model/version identifiers and evaluation configuration.
- JSON artifacts and generated reports are stored together for audit.
