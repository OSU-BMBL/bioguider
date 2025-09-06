from __future__ import annotations

from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.agents.common_conversation import CommonConversation


CLEANUP_PROMPT = """
You are “BioGuider,” a precise editor for biomedical/bioinformatics documentation.

TASK
Given a full README markdown, produce a corrected version that:
- Fixes typos, grammar, capitalization, and spacing
- Corrects malformed markdown (headers, lists, links, code fences)
- Repairs or normalizes link formatting; keep URLs absolute if present
- Removes duplicated sections or repeated content; consolidate if needed
- Preserves technical accuracy and biomedical domain terminology (do not invent features)
- Keeps tone neutral and professional; avoid marketing language
- Preserves all valid information; do not delete content unless it is a duplicate or malformed

INPUT
<<README>>
{readme}
<</README>>

OUTPUT
Return ONLY the revised markdown (no commentary, no explanations).
"""


class LLMCleaner:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def clean_readme(self, content: str) -> tuple[str, dict]:
        conv = CommonConversation(self.llm)
        output, token_usage = conv.generate(
            system_prompt=CLEANUP_PROMPT.format(readme=content[:30000]),
            instruction_prompt="Provide the corrected README markdown only.",
        )
        return output.strip(), token_usage


