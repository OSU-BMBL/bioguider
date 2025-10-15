from __future__ import annotations

from typing import Dict
import json
from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.agents.common_conversation import CommonConversation
from .models import StyleProfile, SuggestionItem


LLM_SECTION_PROMPT = """
You are "BioGuider," a precise documentation generator for biomedical/bioinformatics software.

GOAL
Write or refine a single documentation section named "{section}". Follow the specific guidance from the evaluation report exactly.

INPUTS (use only what is provided; never invent)
- suggestion_category: {suggestion_category}
- anchor_title: {anchor_title}
- guidance: {guidance}
- evidence_from_evaluation: {evidence}
- repo_context_excerpt (analyze tone/formatting; do not paraphrase it blindly): <<{context}>>

CRITICAL REQUIREMENTS
- Follow the guidance EXACTLY as provided: {guidance}
- Address the specific suggestions from the evaluation report precisely
- Do not deviate from the guidance or add unrelated content
- If guidance mentions specific packages, requirements, or details, include them exactly
- For RMarkdown files (.Rmd), preserve the original structure including YAML frontmatter, code chunks, and existing headers
- NEVER generate generic placeholder content like "Clear 2–3 sentence summary" or "brief description"
- ABSOLUTELY FORBIDDEN: Do NOT add summary sections, notes, conclusions, or any text at the end of documents
- ABSOLUTELY FORBIDDEN: Do NOT wrap content in markdown code fences (```markdown). Return pure content only.
- ABSOLUTELY FORBIDDEN: Do NOT add phrases like "Happy analyzing!", "Ensure all dependencies are up-to-date", or any concluding statements
- ALWAYS use the specific guidance provided above to create concrete, actionable content

STYLE & CONSTRAINTS
- Fix obvious errors in the content.
- Preserve the existing tone and style markers: {tone_markers}
- Use heading style "{heading_style}" and list style "{list_style}"; link style "{link_style}".
- Neutral, professional tone; avoid marketing claims.
- Omit details you cannot substantiate from inputs/context; do not invent.
- Prefer bullets; keep it short and skimmable.
- Biomedical examples must avoid PHI; assume de-identified data.
- Output must be plain markdown for this section only, with no commentary and no backticks.
- Avoid duplication: if similar content exists in the repo context, rewrite succinctly instead of repeating.
- Never remove, alter, or recreate top-of-file badges/shields/logos (e.g., CI, PyPI, Conda, Docs shields). Assume they remain unchanged; do not output replacements for them.
- When targeting README content, do not rewrite the document title or header area; generate only the requested section body to be inserted below existing headers/badges.

SECTION GUIDELINES (follow guidance exactly)
- Dependencies: Include specific packages mentioned in guidance (e.g., "ggplot2", "dplyr", etc.)
- System Requirements: Include R version requirements and platform-specific instructions as mentioned in guidance
- Hardware Requirements: Include RAM/CPU recommendations as specified in guidance
- License: one sentence referencing the license and pointing to the LICENSE file.
- Install (clarify dependencies): Include compatibility details across operating systems and architectures as mentioned in guidance
- Tutorial improvements: Add specific examples, error handling, and reproducibility notes as mentioned in guidance
- User guide improvements: Enhance clarity, add missing information, and improve error handling as mentioned in guidance
- Conservative injection: For tutorial files (.Rmd), make minimal, targeted additions that preserve the original structure and flow. Add brief notes, small subsections, or contextual comments that enhance existing content without disrupting the tutorial's narrative.
- RMarkdown integration: When inserting content into existing RMarkdown tutorials, integrate naturally into the flow rather than creating standalone sections. Add brief explanatory text, code comments, or small subsections that enhance the existing content.
- RMarkdown format compliance: For .Rmd files, ensure content follows RMarkdown conventions:
  * Use proper R code chunks with ```{{r chunk_name}} and ``` when adding code examples
  * Maintain the tutorial's existing tone and context - content should feel like a natural continuation
  * Avoid creating new major sections unless absolutely necessary
  * Use inline R code with `{{r code_here}}` when appropriate
  * Keep explanations concise and contextual to the tutorial's purpose
- Context awareness: Content should feel like a natural part of the existing tutorial, not a standalone addition. Reference the tutorial's specific context, datasets, and examples.
- If the section does not fit the above, produce content that directly addresses the guidance provided.

OUTPUT FORMAT
- Return only the section markdown (no code fences).
- Start with a level-2 header: "## {{anchor_title}}" unless the content already starts with a header.
- Ensure the content directly addresses: {{guidance}}
- DO NOT include generic instructions or placeholder text
- ONLY generate content that fulfills the specific guidance provided
"""

LLM_FULLDOC_PROMPT = """
You are "BioGuider," a documentation rewriter.

GOAL
Rewrite a complete target document using only the provided evaluation report signals and the repository context excerpts. Output a full, ready-to-publish markdown file that is more complete and directly usable.

INPUTS (authoritative)
- evaluation_report (structured JSON excerpts): <<{evaluation_report}>>
- target_file: {target_file}
- repo_context_excerpt (do not copy blindly; use only to keep style/tone): <<{context}>>

STRICT CONSTRAINTS
- Base the content solely on the evaluation report. Do not invent features, data, or claims not supported by it.
- Prefer completeness and usability: produce the full file content, not just minimal "added" snippets.
- Preserve top-of-file badges/logos if they exist in the original; keep title and header area intact unless the report requires changes.
- CRITICAL: Preserve the original document structure, sections, and flow. Only enhance existing content and add missing information.
- For tutorial files (.Rmd), maintain all original sections (Docker, installation methods, etc.) while improving clarity and adding missing details.
- Fix obvious errors; improve structure and readability per report suggestions.
- Include ONLY sections specifically requested by the evaluation report - do not add unnecessary sections.
- Avoid redundancy: do not duplicate information across multiple sections.
- ABSOLUTELY FORBIDDEN: Do NOT add summary sections, notes, conclusions, or any text at the end of documents
- ABSOLUTELY FORBIDDEN: Do NOT wrap the entire document inside markdown code fences (```markdown). Do NOT start with ```markdown or end with ```. Return pure markdown content suitable for copy/paste.
- ABSOLUTELY FORBIDDEN: Do NOT add phrases like "Happy analyzing!" or any concluding statements
- Keep links well-formed; keep neutral, professional tone; concise, skimmable formatting.
- For RMarkdown files (.Rmd), preserve YAML frontmatter exactly and do not wrap content in code fences.

OUTPUT
- Return only the full markdown content for {target_file}. No commentary, no fences.
"""

LLM_README_COMPREHENSIVE_PROMPT = """
You are "BioGuider," a comprehensive documentation rewriter specializing in README files.

GOAL
Create a complete, professional README.md that addresses all evaluation suggestions comprehensively. This is the main project documentation that users will see first.

INPUTS (authoritative)
- evaluation_report (structured JSON excerpts): <<{evaluation_report}>>
- target_file: {target_file}
- repo_context_excerpt (do not copy blindly; use only to keep style/tone): <<{context}>>

COMPREHENSIVE README REQUIREMENTS
- Create a complete README with all essential sections: Overview, Installation, Usage, Examples, Contributing, License
- Address ALL evaluation suggestions thoroughly and comprehensively
- Include detailed dependency information with installation commands
- Provide clear system requirements and compatibility information
- Add practical usage examples and code snippets
- Include troubleshooting section if needed
- Make it copy-paste ready for users
- Use professional, clear language suitable for biomedical researchers

STRICT CONSTRAINTS
- Base the content solely on the evaluation report. Do not invent features, data, or claims not supported by it.
- ABSOLUTELY FORBIDDEN: Do NOT wrap the entire document inside markdown code fences (```markdown). Return pure markdown content.
- ABSOLUTELY FORBIDDEN: Do NOT add summary sections, notes, conclusions, or any text at the end of documents
- Keep links well-formed; use neutral, professional tone; concise, skimmable formatting.

OUTPUT
- Return only the full README.md content. No commentary, no fences.
"""


class LLMContentGenerator:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def generate_section(self, suggestion: SuggestionItem, style: StyleProfile, context: str = "") -> tuple[str, dict]:
        conv = CommonConversation(self.llm)
        section_name = suggestion.anchor_hint or suggestion.category.split(".")[-1].replace("_", " ").title()
        system_prompt = LLM_SECTION_PROMPT.format(
            tone_markers=", ".join(style.tone_markers or []),
            heading_style=style.heading_style,
            list_style=style.list_style,
            link_style=style.link_style,
            section=section_name,
            anchor_title=section_name,
            suggestion_category=suggestion.category,
            evidence=(suggestion.source.get("evidence", "") if suggestion.source else ""),
            context=context[:2500],
            guidance=(suggestion.content_guidance or "").strip(),
        )
        content, token_usage = conv.generate(system_prompt=system_prompt, instruction_prompt="Write the section content now.")
        return content.strip(), token_usage

    def generate_full_document(self, target_file: str, evaluation_report: dict, context: str = "") -> tuple[str, dict]:
        conv = CommonConversation(self.llm)
        
        # Use comprehensive README prompt for README.md files
        if target_file.endswith("README.md"):
            system_prompt = LLM_README_COMPREHENSIVE_PROMPT.format(
                target_file=target_file,
                evaluation_report=json.dumps(evaluation_report)[:6000],
                context=context[:4000],
            )
        else:
            system_prompt = LLM_FULLDOC_PROMPT.format(
                target_file=target_file,
                evaluation_report=json.dumps(evaluation_report)[:6000],
                context=context[:4000],
            )
        
        content, token_usage = conv.generate(system_prompt=system_prompt, instruction_prompt="Write the full document now.")
        return content.strip(), token_usage


