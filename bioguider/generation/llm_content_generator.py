from __future__ import annotations

from typing import Dict, Optional
import json
import re
import os
from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.agents.common_conversation import CommonConversation
from .models import StyleProfile, SuggestionItem
from .prompts import PromptLoader, PromptTemplate, format_prompt
from .truncation_handler import TruncationHandler
from .rmarkdown_processor import RMarkdownProcessor, ChunkType


class LLMContentGenerator:
    """
    Generates documentation content using LLMs.

    Uses extracted modules for:
    - Prompt templates (PromptLoader)
    - Truncation detection (TruncationHandler)
    - RMarkdown processing (RMarkdownProcessor)
    """

    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm
        self.prompt_loader = PromptLoader()
        self.truncation_handler = TruncationHandler()
        self.rmarkdown_processor = RMarkdownProcessor()

    def _detect_truncation(
        self, content: str, target_file: str, original_content: str = None
    ) -> bool:
        """
        Detect if content appears to be truncated.

        Delegates to TruncationHandler for actual detection logic.

        Args:
            content: Generated content to check
            target_file: Target file path for context
            original_content: Original content for comparison (if available)

        Returns:
            True if content appears truncated, False otherwise
        """
        return self.truncation_handler.is_truncated(
            content, target_file, original_content
        )

    def _find_continuation_point(
        self, content: str, original_content: str = None
    ) -> Optional[str]:
        """
        Find a suitable continuation point in the content.

        Delegates to TruncationHandler for actual logic.

        Args:
            content: The generated content so far
            original_content: The original content for comparison

        Returns:
            A suitable continuation point, or None if not found
        """
        return self.truncation_handler.find_continuation_point(
            content, original_content
        )

    def _appears_complete(
        self, content: str, target_file: str, original_content: str = None
    ) -> bool:
        """
        Check if content appears to be complete.

        Delegates to TruncationHandler for actual detection logic.

        Args:
            content: Generated content to check
            target_file: Target file path for context
            original_content: Original content for length comparison (optional but recommended)

        Returns:
            True if content appears complete, False if it needs continuation
        """
        return self.truncation_handler.is_complete(
            content, target_file, original_content
        )

    def _generate_continuation(
        self,
        target_file: str,
        evaluation_report: dict,
        context: str,
        existing_content: str,
    ) -> tuple[str, dict]:
        """
        Generate continuation content from where previous generation left off.

        Args:
            target_file: Target file path
            evaluation_report: Evaluation report data
            context: Repository context
            existing_content: Previously generated content

        Returns:
            Tuple of (continuation_content, token_usage)
        """
        # Create LLM for continuation (uses 16k tokens by default)
        from bioguider.agents.agent_utils import get_llm
        import os

        llm = get_llm(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
            azure_deployment=os.environ.get("OPENAI_DEPLOYMENT_NAME"),
        )

        conv = CommonConversation(llm)

        # Calculate total suggestions for the prompt
        total_suggestions = 1
        if isinstance(evaluation_report, dict):
            if "total_suggestions" in evaluation_report:
                total_suggestions = evaluation_report["total_suggestions"]
            elif "suggestions" in evaluation_report and isinstance(
                evaluation_report["suggestions"], list
            ):
                total_suggestions = len(evaluation_report["suggestions"])

        # Use the centralized continuation prompt template from PromptLoader
        continuation_prompt = self.prompt_loader.format(
            PromptTemplate.CONTINUATION,
            existing_content_tail=existing_content[
                -1000:
            ],  # Last 1000 chars for context
        )

        content, token_usage = conv.generate(
            system_prompt=continuation_prompt,
            instruction_prompt="Continue the document from where it left off.",
        )
        return content.strip(), token_usage

    def generate_section(
        self, suggestion: SuggestionItem, style: StyleProfile, context: str = ""
    ) -> tuple[str, dict]:
        conv = CommonConversation(self.llm)
        section_name = (
            suggestion.anchor_hint
            or suggestion.category.split(".")[-1].replace("_", " ").title()
        )

        # Extract original text snippet and evaluation score from suggestion source
        original_text = ""
        evaluation_score = ""
        if hasattr(suggestion, "source") and suggestion.source:
            original_text = suggestion.source.get("original_text", "")
            evaluation_score = suggestion.source.get("score", "")

        # Detect document context to help with appropriate responses
        document_context = self._detect_document_context(
            context, suggestion.anchor_title or ""
        )

        system_prompt = self.prompt_loader.format(
            PromptTemplate.SECTION,
            tone_markers=", ".join(style.tone_markers or []),
            heading_style=style.heading_style,
            list_style=style.list_style,
            link_style=style.link_style,
            section=section_name,
            anchor_title=section_name,
            suggestion_category=suggestion.category,
            original_text=original_text,
            evaluation_score=evaluation_score,
            context=context[:2500],
            guidance=(suggestion.content_guidance or "").strip(),
        )

        # Add context-aware instruction
        context_instruction = f"\n\nCONTEXT DETECTED: {document_context}\n"
        if document_context == "TUTORIAL":
            context_instruction += "Focus on usage/analysis steps, NOT installation. Users already have software installed.\n"
        elif document_context == "README":
            context_instruction += "Focus on installation, setup, and getting started. Users need to install software.\n"
        elif document_context == "BIOLOGICAL":
            context_instruction += "Use accurate biological terminology and provide biologically meaningful examples.\n"

        system_prompt += context_instruction
        content, token_usage = conv.generate(
            system_prompt=system_prompt,
            instruction_prompt="Write the section content now.",
        )
        return content.strip(), token_usage

    def generate_full_document(
        self,
        target_file: str,
        evaluation_report: dict,
        context: str = "",
        original_content: str = None,
    ) -> tuple[str, dict]:
        # Create LLM (uses 16k tokens by default - enough for any document)
        from bioguider.agents.agent_utils import get_llm
        import os
        import json
        from datetime import datetime

        # Get LLM with default 16k token limit
        llm = get_llm(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
            azure_deployment=os.environ.get("OPENAI_DEPLOYMENT_NAME"),
        )

        conv = CommonConversation(llm)

        # Debug: Save generation settings and context
        debug_info = {
            "target_file": target_file,
            "timestamp": datetime.now().isoformat(),
            "evaluation_report": evaluation_report,
            "context_length": len(context),
            "llm_settings": {
                "model_name": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                "azure_deployment": os.environ.get("OPENAI_DEPLOYMENT_NAME"),
                "max_tokens": getattr(llm, "max_tokens", 16384),
            },
        }

        # Save debug info to file
        debug_dir = "outputs/debug_generation"
        os.makedirs(debug_dir, exist_ok=True)
        safe_filename = target_file.replace("/", "_").replace(".", "_")
        debug_file = os.path.join(debug_dir, f"{safe_filename}_debug.json")
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(debug_info, f, indent=2, ensure_ascii=False)

        # Debug: Save raw evaluation_report to see what's being serialized
        eval_report_file = os.path.join(
            debug_dir, f"{safe_filename}_raw_eval_report.json"
        )
        with open(eval_report_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

        # Use comprehensive README prompt for README.md files
        if target_file.endswith("README.md"):
            system_prompt = self.prompt_loader.format(
                PromptTemplate.README_COMPREHENSIVE,
                target_file=target_file,
                evaluation_report=json.dumps(evaluation_report)[:6000],
                context=context[:4000],
                original_content=original_content or "",
            )
        else:
            # Calculate total suggestions for the prompt
            total_suggestions = 1
            if isinstance(evaluation_report, dict):
                if "total_suggestions" in evaluation_report:
                    total_suggestions = evaluation_report["total_suggestions"]
                elif "suggestions" in evaluation_report and isinstance(
                    evaluation_report["suggestions"], list
                ):
                    total_suggestions = len(evaluation_report["suggestions"])

            system_prompt = self.prompt_loader.format(
                PromptTemplate.FULL_DOCUMENT,
                target_file=target_file,
                evaluation_report=json.dumps(evaluation_report)[:6000],
                context=context[:4000],
                original_content=original_content or "",
                total_suggestions=total_suggestions,
            )

        # Save initial prompt for debugging
        prompt_file = os.path.join(debug_dir, f"{safe_filename}_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write("=== SYSTEM PROMPT ===\n")
            f.write(system_prompt)
            f.write("\n\n=== INSTRUCTION PROMPT ===\n")
            f.write("Write the full document now.")
            # Context is already embedded in system prompt; avoid duplicating here

        # Initial generation
        # If the original document is long (RMarkdown > 8k chars), avoid truncation by chunked rewrite
        # Lower threshold from 12k to 8k to catch more documents that would otherwise truncate
        use_chunked = bool(
            target_file.endswith(".Rmd")
            and isinstance(original_content, str)
            and len(original_content) > 8000
        )
        if use_chunked:
            content, token_usage = self._generate_full_document_chunked(
                target_file=target_file,
                evaluation_report=evaluation_report,
                context=context,
                original_content=original_content or "",
                debug_dir=debug_dir,
                safe_filename=safe_filename,
            )
        else:
            content, token_usage = conv.generate(
                system_prompt=system_prompt,
                instruction_prompt="Write the full document now.",
            )
            content = content.strip()

        # Save initial generation for debugging
        generation_file = os.path.join(debug_dir, f"{safe_filename}_generation_0.txt")
        with open(generation_file, "w", encoding="utf-8") as f:
            f.write(f"=== INITIAL GENERATION ===\n")
            f.write(f"Tokens: {token_usage}\n")
            f.write(f"Length: {len(content)} characters\n")
            if original_content:
                f.write(f"Original length: {len(original_content)} characters\n")
            f.write(
                f"Truncation detected: {self._detect_truncation(content, target_file, original_content)}\n"
            )
            f.write(f"\n=== CONTENT ===\n")
            f.write(content)

        # Check for truncation and continue if needed
        max_continuations = 3  # Limit to prevent infinite loops
        continuation_count = 0

        while (
            not use_chunked
            and self._detect_truncation(content, target_file, original_content)
            and continuation_count < max_continuations
        ):
            # Additional check: if content appears complete, don't continue
            # Pass original_content so we can check length ratio
            if self._appears_complete(content, target_file, original_content):
                break
            continuation_count += 1

            # Calculate total suggestions for debugging info
            total_suggestions = 1
            if isinstance(evaluation_report, dict):
                if "total_suggestions" in evaluation_report:
                    total_suggestions = evaluation_report["total_suggestions"]
                elif "suggestions" in evaluation_report and isinstance(
                    evaluation_report["suggestions"], list
                ):
                    total_suggestions = len(evaluation_report["suggestions"])

            # Find better continuation point - look for last complete section
            continuation_point = self._find_continuation_point(
                content, original_content
            )
            if not continuation_point:
                continuation_point = content[-1000:]  # Fallback to last 1000 chars

            # Generate continuation prompt using centralized template
            continuation_prompt = self.prompt_loader.format(
                PromptTemplate.CONTINUATION,
                existing_content_tail=continuation_point,
            )

            # Save continuation prompt for debugging
            continuation_prompt_file = os.path.join(
                debug_dir,
                f"{safe_filename}_continuation_{continuation_count}_prompt.txt",
            )
            with open(continuation_prompt_file, "w", encoding="utf-8") as f:
                f.write(continuation_prompt)

            # Generate continuation
            continuation_content, continuation_usage = self._generate_continuation(
                target_file=target_file,
                evaluation_report=evaluation_report,
                context=context,
                existing_content=content,
            )

            # Save continuation generation for debugging
            continuation_file = os.path.join(
                debug_dir, f"{safe_filename}_continuation_{continuation_count}.txt"
            )
            with open(continuation_file, "w", encoding="utf-8") as f:
                f.write(f"=== CONTINUATION {continuation_count} ===\n")
                f.write(f"Tokens: {continuation_usage}\n")
                f.write(f"Length: {len(continuation_content)} characters\n")
                f.write(
                    f"Truncation detected: {self._detect_truncation(continuation_content, target_file)}\n"
                )
                f.write(f"\n=== CONTENT ===\n")
                f.write(continuation_content)

            # Merge continuation with existing content
            if continuation_content:
                content += "\n\n" + continuation_content
                # Update token usage
                token_usage = {
                    "total_tokens": token_usage.get("total_tokens", 0)
                    + continuation_usage.get("total_tokens", 0),
                    "prompt_tokens": token_usage.get("prompt_tokens", 0)
                    + continuation_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0)
                    + continuation_usage.get("completion_tokens", 0),
                }

                # Save merged content for debugging
                merged_file = os.path.join(
                    debug_dir, f"{safe_filename}_merged_{continuation_count}.txt"
                )
                with open(merged_file, "w", encoding="utf-8") as f:
                    f.write(
                        f"=== MERGED CONTENT AFTER CONTINUATION {continuation_count} ===\n"
                    )
                    f.write(f"Total length: {len(content)} characters\n")
                    f.write(
                        f"Truncation detected: {self._detect_truncation(content, target_file)}\n"
                    )
                    f.write(f"\n=== CONTENT ===\n")
                    f.write(content)
            else:
                # If continuation is empty, break to avoid infinite loop
                break

        # Clean up any markdown code fences that might have been added
        content = self._clean_markdown_fences(content)

        # Save final cleaned content for debugging
        final_file = os.path.join(debug_dir, f"{safe_filename}_final.txt")
        with open(final_file, "w", encoding="utf-8") as f:
            f.write(f"=== FINAL CLEANED CONTENT ===\n")
            f.write(f"Total tokens: {token_usage}\n")
            f.write(f"Final length: {len(content)} characters\n")
            f.write(f"Continuations used: {continuation_count}\n")
            f.write(f"\n=== CONTENT ===\n")
            f.write(content)

        return content, token_usage

    def _clean_markdown_fences(self, content: str) -> str:
        """
        Remove markdown code fences that shouldn't be in the final content.
        """
        # Remove ```markdown at the beginning
        if content.startswith("```markdown\n"):
            content = content[12:]  # Remove ```markdown\n

        # Remove ``` at the end
        if content.endswith("\n```"):
            content = content[:-4]  # Remove \n```
        elif content.endswith("```"):
            content = content[:-3]  # Remove ```

        # Remove any standalone ```markdown lines
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip() == "```markdown":
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _split_rmd_into_chunks(self, content: str) -> list[dict]:
        """
        Split RMarkdown content into chunks for processing.

        Delegates to RMarkdownProcessor for actual chunking logic.

        Returns list of dicts with 'type' (yaml/code/text) and 'content'.
        """
        return self.rmarkdown_processor.split_into_chunks_legacy(content)

    def _generate_text_chunk(
        self,
        conv: CommonConversation,
        evaluation_report: dict,
        context: str,
        chunk_text: str,
    ) -> tuple[str, dict]:
        LLM_CHUNK_PROMPT = (
            "You are BioGuider improving a single TEXT chunk of a larger RMarkdown document.\n\n"
            "GOAL\nRefine ONLY the given chunk's prose per evaluation suggestions while preserving structure.\n"
            "Do not add conclusions or new sections.\n\n"
            "INPUTS\n- evaluation_report: <<{evaluation_report}>>\n- repo_context_excerpt: <<{context}>>\n- original_chunk:\n<<<\n{chunk}\n>>>\n\n"
            "CRITICAL RULES\n"
            "- This is a TEXT-ONLY chunk - do NOT add any code blocks or code fences (```).\n"
            "- Preserve all headers and formatting in this chunk.\n"
            "- Do not invent technical specs.\n"
            "- Output ONLY the refined text (no code fences, no markdown code blocks).\n"
            "- NEVER add ``` anywhere in your output.\n"
            "- Keep the same approximate length as the original chunk."
        )
        system_prompt = LLM_CHUNK_PROMPT.format(
            evaluation_report=json.dumps(evaluation_report)[:4000],
            context=context[:1500],
            chunk=chunk_text[:6000],
        )
        content, usage = conv.generate(
            system_prompt=system_prompt,
            instruction_prompt="Rewrite this text chunk now. Remember: NO code fences (```).",
        )

        # Post-processing: remove any code fences that may have been added
        output = content.strip()

        # If output contains code fences, the LLM didn't follow instructions
        # Return original to preserve document structure
        if "```" in output:
            print(f"WARNING: LLM added code fences to text chunk, using original")
            return chunk_text, usage

        return output, usage

    def _generate_full_document_chunked(
        self,
        target_file: str,
        evaluation_report: dict,
        context: str,
        original_content: str,
        debug_dir: str,
        safe_filename: str,
    ) -> tuple[str, dict]:
        conv = CommonConversation(self.llm)
        chunks = self._split_rmd_into_chunks(original_content)
        merged = []
        total_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        from datetime import datetime

        # Save chunk analysis for debugging
        chunk_analysis_file = os.path.join(
            debug_dir, f"{safe_filename}_chunk_analysis.txt"
        )
        with open(chunk_analysis_file, "w", encoding="utf-8") as f:
            f.write(f"Total chunks: {len(chunks)}\n")
            for idx, ch in enumerate(chunks):
                f.write(
                    f"Chunk {idx}: type={ch['type']}, length={len(ch['content'])}\n"
                )
                if ch["type"] == "code":
                    f.write(f"  First line: {ch['content'].split(chr(10))[0][:80]}\n")

        for idx, ch in enumerate(chunks):
            if ch["type"] in ("yaml", "code"):
                # CRITICAL: Pass through code/yaml chunks EXACTLY as-is
                merged.append(ch["content"])
                continue

            # For text chunks, try to improve but fall back to original if needed
            out, usage = self._generate_text_chunk(
                conv, evaluation_report, context, ch["content"]
            )

            # Validate the output doesn't contain code fence fragments that could break structure
            if not out or "```" in out:
                # If LLM added code fences in text chunk, it could break the document
                # Fall back to original text
                out = ch["content"]

            merged.append(out)
            try:
                total_usage["total_tokens"] += int(usage.get("total_tokens", 0))
                total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                total_usage["completion_tokens"] += int(
                    usage.get("completion_tokens", 0)
                )
            except Exception:
                pass
            chunk_file = os.path.join(debug_dir, f"{safe_filename}_chunk_{idx}.txt")
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(
                    f"=== CHUNK {idx} ({ch['type']}) at {datetime.now().isoformat()} ===\n"
                )
                f.write(out)

        content = "\n".join(merged)

        # CRITICAL: Validate code block structure is preserved
        original_fences = len(re.findall(r"^```", original_content, flags=re.M))
        generated_fences = len(re.findall(r"^```", content, flags=re.M))

        if original_fences != generated_fences:
            # Code block structure was broken - log error and return original
            error_file = os.path.join(
                debug_dir, f"{safe_filename}_ERROR_codeblock_mismatch.txt"
            )
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"ERROR: Code block count mismatch!\n")
                f.write(f"Original: {original_fences} code fences\n")
                f.write(f"Generated: {generated_fences} code fences\n")
                f.write(f"\nReturning original content to preserve structure.\n")
            print(
                f"WARNING: Code block structure broken for {target_file}, returning original content"
            )
            return original_content, total_usage

        return content, total_usage

    def _detect_document_context(self, context: str, anchor_title: str) -> str:
        """Detect the document context to help with appropriate responses."""
        context_lower = context.lower()
        anchor_lower = anchor_title.lower()

        # Check for tutorial context
        if any(
            keyword in context_lower
            for keyword in [
                "tutorial",
                "vignette",
                "example",
                "workflow",
                "step-by-step",
            ]
        ):
            return "TUTORIAL"

        # Check for README context
        if any(
            keyword in context_lower
            for keyword in ["readme", "installation", "setup", "prerequisites"]
        ):
            return "README"

        # Check for documentation context
        if any(
            keyword in context_lower
            for keyword in ["documentation", "guide", "manual", "reference"]
        ):
            return "DOCUMENTATION"

        # Check for biological context
        if any(
            keyword in context_lower
            for keyword in [
                "cell",
                "gene",
                "protein",
                "dna",
                "rna",
                "genome",
                "transcriptome",
                "proteome",
                "metabolome",
            ]
        ):
            return "BIOLOGICAL"

        # Default to general context
        return "GENERAL"
