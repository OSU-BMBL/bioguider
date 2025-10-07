from __future__ import annotations

from typing import Dict
from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.agents.common_conversation import CommonConversation
from .models import StyleProfile, SuggestionItem


LLM_SECTION_PROMPT = """
You are “BioGuider,” a concise documentation generator for biomedical/bioinformatics software.

GOAL
Write or refine a single documentation section named "{section}". Produce professional, comprehensive, style-consistent content that addresses only this section.

INPUTS (use only what is provided; never invent)
- suggestion_category: {suggestion_category}
- anchor_title: {anchor_title}
- guidance: {guidance}
- evidence_from_evaluation: {evidence}
- repo_context_excerpt (analyze tone/formatting; do not paraphrase it blindly): <<{context}>>

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

SECTION GUIDELINES
- Dependencies: short bullet list; clearly separate Mandatory and Optional if applicable.
- System Requirements: runtime versions and supported OS; add hardware notes only if guidance provides specifics.
- Hardware Requirements: brief bullets with RAM/CPU only if guidance includes numbers.
- License: one sentence referencing the license and pointing to the LICENSE file.
- Install (clarify dependencies): bullets under Mandatory and Optional.
- If the section does not fit the above, produce a concise, accurate subsection aligned with the repo’s style.

OUTPUT FORMAT
- Return only the section markdown (no code fences).
- Start with a level-2 header: "## {anchor_title}" unless the content already starts with a header.
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


