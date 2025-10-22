from __future__ import annotations

from typing import Dict
import json
import re
import os
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
- repo_context_excerpt (analyze tone/formatting; do not paraphrase it blindly): <<{context}>>

CRITICAL REQUIREMENTS
- Follow the guidance EXACTLY as provided: {guidance}
- Address the specific suggestions from the evaluation report precisely
- Do not deviate from the guidance or add unrelated content
- If guidance mentions specific packages, requirements, or details, include them ONLY if they are explicitly stated - never invent or estimate
- Preserve the original file structure including frontmatter, code blocks, and existing headers
- NEVER generate generic placeholder content like "Clear 2–3 sentence summary" or "brief description"
- NEVER invent technical specifications (hardware requirements, version numbers, performance metrics) unless explicitly provided in guidance or context
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
- Dependencies: Include ONLY specific packages explicitly mentioned in guidance or found in repo context. Never invent package names or versions.
- System Requirements: Include ONLY language/runtime version requirements explicitly stated in guidance or found in repo context. Never invent version numbers.
- Hardware Requirements: Include ONLY specific RAM/CPU recommendations explicitly stated in guidance or found in repo context. NEVER estimate or invent hardware specifications - omit this section if not substantiated.
- License: one sentence referencing the license and pointing to the LICENSE file.
- Install (clarify dependencies): Include compatibility details ONLY if explicitly mentioned in guidance or found in repo context.
- Tutorial improvements: Add specific examples, error handling, and reproducibility notes as mentioned in guidance
- User guide improvements: Enhance clarity, add missing information, and improve error handling as mentioned in guidance
- Conservative injection: For tutorial files, make minimal, targeted additions that preserve the original structure and flow. Add brief notes, small subsections, or contextual comments that enhance existing content without disrupting the tutorial's narrative.
- Natural integration: When inserting content into existing tutorials or guides, integrate naturally into the flow rather than creating standalone sections. Add brief explanatory text, code comments, or small subsections that enhance the existing content.
- Format compliance: Preserve the existing file format conventions (e.g., YAML frontmatter, code blocks, headers):
  * For code examples, use the appropriate code fence syntax for the language (e.g., ```r, ```python, ```bash)
  * Maintain the tutorial's existing tone and context - content should feel like a natural continuation
  * Avoid creating new major sections unless absolutely necessary
  * Keep explanations concise and contextual to the tutorial's purpose
- Context awareness: Content should feel like a natural part of the existing document, not a standalone addition. Reference the document's specific context, datasets, and examples when available.
- Biological accuracy: For biomedical/bioinformatics content, ensure technical accuracy. If unsure about biological or computational details, keep descriptions general rather than inventing specifics.
- If the section does not fit the above, produce content that directly addresses the guidance provided.

OUTPUT FORMAT
- Return only the section markdown (no code fences).
- Start with a level-2 header: "## {{anchor_title}}" unless the content already starts with a header.
- Ensure the content directly addresses: {{guidance}}
- DO NOT include generic instructions or placeholder text
- ONLY generate content that fulfills the specific guidance provided
"""

LLM_FULLDOC_PROMPT = """
You are "BioGuider," a documentation rewriter with enhanced capabilities for complex documents.

GOAL
Rewrite a complete target document by enhancing the existing content while maintaining the EXACT original structure, sections, and flow. Use only the provided evaluation report signals and repository context excerpts. Output a full, ready-to-publish markdown file that follows the original document structure precisely while incorporating improvements. You now have increased token capacity to handle complex documents comprehensively.

INPUTS (authoritative)
- evaluation_report (structured JSON excerpts): <<{evaluation_report}>>
- target_file: {target_file}
- repo_context_excerpt (do not copy blindly; use only to keep style/tone): <<{context}>>

CRITICAL: SINGLE DOCUMENT WITH MULTIPLE IMPROVEMENTS
This file requires improvements from {total_suggestions} separate evaluation suggestions. You must:
1. **Read ALL {total_suggestions} suggestions** in the evaluation_report before writing
2. **Integrate ALL suggestions into ONE cohesive document** - do NOT create {total_suggestions} separate versions
3. **Weave improvements together naturally** - related suggestions should enhance the same sections
4. **Write the document ONCE** with all improvements incorporated throughout

INTEGRATION STRATEGY
- **CRITICAL**: Follow the EXACT structure of the original document. Do NOT create new sections.
- Identify which suggestions target existing sections in the original document
- Apply improvements ONLY to existing sections - do NOT create new sections
- For tutorial files: Enhance existing sections with relevant suggestions, maintain original section order
- For documentation files: Merge suggestions into existing structure, avoid redundant sections
- Result: ONE enhanced document that follows the original structure and addresses all {total_suggestions} suggestions

CAPACITY AND SCOPE
- You have enhanced token capacity to handle complex documents comprehensively
- Tutorial documents: Enhanced capacity for step-by-step content, code examples, and comprehensive explanations
- Complex documents: Increased capacity for multiple sections, detailed explanations, and extensive content
- Comprehensive documents: Full capacity for complete documentation with all necessary sections

STRICT CONSTRAINTS
- **CRITICAL**: Follow the EXACT structure and sections of the original document. Do NOT create new sections or reorganize content.
- Base the content solely on the evaluation report and repo context. Do not invent features, data, or claims not supported by these sources.
- CRITICAL: NEVER invent technical specifications including:
  * Hardware requirements (RAM, CPU, disk space) unless explicitly stated in guidance/context
  * Version numbers for dependencies unless explicitly stated in guidance/context
  * Performance metrics, benchmarks, or timing estimates
  * Biological/computational parameters or thresholds without substantiation
  * Installation commands or package names not found in the repo context
- **CRITICAL**: Preserve the original document structure, sections, and flow EXACTLY. Only enhance existing content and add missing information based on evaluation suggestions.
- For tutorial files, maintain ALL original sections in their original order while improving clarity and adding missing details based on evaluation suggestions.
- Fix obvious errors; improve structure and readability per report suggestions.
- Include ONLY sections that exist in the original document - do not add unnecessary sections.
- Avoid redundancy: do not duplicate information across multiple sections.
- **ABSOLUTELY CRITICAL**: Do NOT add ANY conclusion, summary, or closing paragraph at the end
- **ABSOLUTELY CRITICAL**: Do NOT wrap the entire document inside markdown code fences (```markdown). Do NOT start with ```markdown or end with ```. Return pure content suitable for copy/paste.
- **ABSOLUTELY CRITICAL**: Do NOT add phrases like "Happy analyzing!", "This vignette demonstrates...", "By following the steps outlined...", or ANY concluding statements
- **ABSOLUTELY CRITICAL**: Stop writing IMMEDIATELY after the last content section from the original document. Do NOT add "## Conclusion", "## Summary", or any final paragraphs
- **CRITICAL**: Do NOT reorganize, rename, or create new sections. Follow the original document structure exactly.
- Keep links well-formed; keep neutral, professional tone; concise, skimmable formatting.
- Preserve file-specific formatting (e.g., YAML frontmatter, code fence syntax) and do not wrap content in extra code fences.

COMPLETENESS REQUIREMENTS
- Generate complete, comprehensive content that addresses all evaluation suggestions
- For complex documents, ensure all sections are fully developed and detailed
- For tutorial documents, include complete step-by-step instructions with examples
- Use the increased token capacity to provide thorough, useful documentation

OUTPUT
- Return only the full markdown content for {target_file}. No commentary, no fences.
"""

LLM_README_COMPREHENSIVE_PROMPT = """
You are "BioGuider," a comprehensive documentation rewriter specializing in README files with enhanced capacity for complex documentation.

GOAL
Create a complete, professional README.md that addresses all evaluation suggestions comprehensively. This is the main project documentation that users will see first. You now have increased token capacity to create thorough, comprehensive documentation.

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

ENHANCED CAPACITY FEATURES
- You have increased token capacity to create comprehensive documentation
- Include detailed explanations, multiple examples, and thorough coverage
- Provide extensive installation instructions with platform-specific details
- Add comprehensive usage examples with different scenarios
- Include detailed API documentation if applicable
- Provide troubleshooting guides with common issues and solutions

STRICT CONSTRAINTS
- Base the content solely on the evaluation report. Do not invent features, data, or claims not supported by it.
- ABSOLUTELY FORBIDDEN: Do NOT wrap the entire document inside markdown code fences (```markdown). Return pure markdown content.
- ABSOLUTELY FORBIDDEN: Do NOT add summary sections, notes, conclusions, or any text at the end of documents
- Keep links well-formed; use neutral, professional tone; concise, skimmable formatting.

COMPLETENESS REQUIREMENTS
- Generate complete, comprehensive content that addresses all evaluation suggestions
- Ensure all sections are fully developed and detailed
- Use the increased token capacity to provide thorough, useful documentation
- Include all necessary information for users to successfully use the software

OUTPUT
- Return only the full README.md content. No commentary, no fences.
"""

# Continuation prompt template - used when document generation is truncated
LLM_CONTINUATION_PROMPT = """
You are "BioGuider," continuing a truncated documentation generation task.

IMPORTANT: This is STRICT CONTINUATION ONLY. You are NOT creating new content.
You are NOT adding conclusions or summaries. You are ONLY completing the missing sections from the original document.

PREVIOUS CONTENT (do not repeat this):
```
{existing_content_tail}
```

STRICT CONTINUATION RULES:
- Examine the previous content above and identify what section it ends with
- Continue IMMEDIATELY after that section with the next missing section from the original document
- Use the EXACT same structure, style, and tone as the existing content
- Add ONLY the specific content that should logically follow from the last section
- Do NOT add ANY conclusions, summaries, additional resources, or wrap-up content
- Do NOT add phrases like "For further guidance", "Additional Resources", or "In conclusion"

MISSING CONTENT TO ADD:
Based on typical RMarkdown vignette structure, if the document ends with "Common Pitfalls", you should add:
- SCT integration example (SCTransform section)
- Session info section
- Details section (if present in original)
- STOP after these sections - do NOT add anything else

CRITICAL: STOP IMMEDIATELY after completing the missing sections from the original document.
Do NOT add "## Additional Resources" or any final sections.

OUTPUT:
- Return ONLY the continuation content that completes the original document structure
- No commentary, no fences, no conclusions, no additional content
"""


class LLMContentGenerator:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def _detect_truncation(self, content: str, target_file: str, original_content: str = None) -> bool:
        """
        Detect if content appears to be truncated based on common patterns.
        Universal detection for all file types.
        
        Args:
            content: Generated content to check
            target_file: Target file path for context
            original_content: Original content for comparison (if available)
            
        Returns:
            True if content appears truncated, False otherwise
        """
        if not content or len(content.strip()) < 100:
            return True
        
        # 1. Compare to original length if available (most reliable indicator)
        if original_content:
            original_len = len(original_content)
            generated_len = len(content)
            # If generated content is significantly shorter than original (< 80%), likely truncated
            if generated_len < original_len * 0.8:
                return True
        
        # 2. Check for very short content (applies to all files)
        # Only flag as truncated if content is very short (< 500 chars)
        if len(content) < 500:
            return True
            
        # 3. Check for incomplete code blocks (any language)
        # Count opening and closing code fences
        code_fence_count = content.count('```')
        if code_fence_count > 0 and code_fence_count % 2 != 0:
            # Unbalanced code fences suggest truncation
            return True
            
        # 4. Check for specific language code blocks
        if target_file.endswith('.Rmd'):
            # R chunks should be complete
            r_chunks_open = re.findall(r'```\{r[^}]*\}', content)
            if r_chunks_open and not content.rstrip().endswith('```'):
                # Has R chunks but doesn't end with closing fence
                return True
        
        if target_file.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
            # Check for incomplete class/function definitions
            lines = content.split('\n')
            last_lines = [line.strip() for line in lines[-5:] if line.strip()]
            if last_lines:
                last_line = last_lines[-1]
                if (last_line.endswith(':') or 
                    last_line.endswith('{') or
                    last_line.endswith('(') or
                    'def ' in last_line or
                    'class ' in last_line or
                    'function ' in last_line):
                    return True
                    
        # 4. Check for incomplete markdown sections (applies to all markdown-like files)
        if any(target_file.endswith(ext) for ext in ['.md', '.Rmd', '.rst', '.txt']):
            lines = content.split('\n')
            last_non_empty_line = None
            for line in reversed(lines):
                if line.strip():
                    last_non_empty_line = line.strip()
                    break
            
            if last_non_empty_line:
                # Check if last line looks incomplete
                incomplete_endings = [
                    '##',   # Header without content
                    '###',  # Header without content  
                    '####', # Header without content
                    '-',    # List item
                    '*',    # List item or emphasis
                    ':',    # Definition or label
                    '|',    # Table row
                ]
                
                for ending in incomplete_endings:
                    if last_non_empty_line.endswith(ending):
                        return True
                        
                # Check if ends with incomplete patterns
                content_end = content[-300:].strip().lower()
                incomplete_patterns = [
                    '## ',      # Section header without content
                    '### ',     # Subsection without content
                    '#### ',    # Sub-subsection without content
                    '```{',     # Incomplete code chunk
                    '```r',     # Incomplete R chunk
                    '```python',# Incomplete Python chunk
                ]
                
                for pattern in incomplete_patterns:
                    if content_end.endswith(pattern.lower()):
                        return True
                    
        return False

    def _find_continuation_point(self, content: str, original_content: str = None) -> str:
        """
        Find a better continuation point than just the last 1000 characters.
        Looks for the last complete section or code block to continue from.

        Args:
            content: The generated content so far
            original_content: The original content for comparison

        Returns:
            A suitable continuation point, or None if not found
        """
        if not content:
            return None

        lines = content.split('\n')
        if len(lines) < 10:  # Too short to find good continuation point
            return None

        # Strategy 1: Find the last complete section (header with content after it)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('## ') and i + 1 < len(lines):
                # Check if there's content after this header
                next_lines = []
                for j in range(i + 1, min(i + 10, len(lines))):  # Look at next 10 lines
                    if lines[j].strip() and not lines[j].strip().startswith('##'):
                        next_lines.append(lines[j])
                    else:
                        break

                if next_lines:  # Found header with content after it
                    # Return from this header onwards
                    return '\n'.join(lines[i:])

        # Strategy 2: Find the last complete code block
        in_code_block = False
        code_block_start = -1

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('```') and not in_code_block:
                in_code_block = True
                code_block_start = i
            elif line.startswith('```') and in_code_block:
                # Found complete code block
                return '\n'.join(lines[code_block_start:])

        # Strategy 3: Find last complete paragraph (ends with period)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and line.endswith('.') and not line.startswith('#') and not line.startswith('```'):
                # Found a complete sentence, return from there
                return '\n'.join(lines[i:])

        # Strategy 4: If original content is available, find where the generated content diverges
        if original_content:
            # Simple approach: find the longest common suffix
            min_len = min(len(content), len(original_content))
            common_length = 0

            for i in range(1, min_len + 1):
                if content[-i:] == original_content[-i:]:
                    common_length = i
                else:
                    break

            if common_length > 100:  # Found significant common ending
                return content[-(common_length + 100):]  # Include some context

        return None

    def _appears_complete(self, content: str, target_file: str, original_content: str = None) -> bool:
        """
        Check if content appears to be complete based on structure, patterns, AND original length.
        Universal completion check for all file types.
        
        CRITICAL: If original_content is provided, generated content MUST be at least 90% of original length
        to be considered complete, regardless of other heuristics. This prevents the LLM from fooling us
        with fake conclusions.
        
        Args:
            content: Generated content to check
            target_file: Target file path for context
            original_content: Original content for length comparison (optional but recommended)
            
        Returns:
            True if content appears complete, False if it needs continuation
        """
        if not content or len(content.strip()) < 100:
            return False
        
        # CRITICAL: If original content is provided, check length ratio first
        # This prevents the LLM from fooling us with fake conclusions
        if original_content and isinstance(original_content, str):
            generated_len = len(content)
            original_len = len(original_content)
            if generated_len < original_len * 0.9:
                # Generated content is too short compared to original - NOT complete
                return False
        
        # 1. Check for balanced code blocks (applies to all files)
        code_block_count = content.count('```')
        if code_block_count > 0 and code_block_count % 2 != 0:
            # Unbalanced code blocks suggest incomplete
            return False
            
        # 2. File type specific checks
        
        # RMarkdown files
        if target_file.endswith('.Rmd'):
            # Check for proper YAML frontmatter
            if not content.startswith('---'):
                return False
                
            # Check for conclusion patterns
            conclusion_patterns = [
                'sessionInfo()',
                'session.info()',
                '## Conclusion',
                '## Summary',
                '## Session Info',
                '</details>',
                'knitr::knit(',
            ]
            
            content_lower = content.lower()
            has_conclusion = any(pattern.lower() in content_lower for pattern in conclusion_patterns)
            
            # If we have a conclusion and balanced code blocks, likely complete
            if has_conclusion and code_block_count > 0:
                return True
        
        # Markdown files
        if target_file.endswith('.md'):
            # Check for conclusion sections
            conclusion_patterns = [
                '## Conclusion',
                '## Summary',
                '## Next Steps',
                '## Further Reading',
                '## References',
                '## License',
            ]
            
            content_lower = content.lower()
            has_conclusion = any(pattern.lower() in content_lower for pattern in conclusion_patterns)
            
            if has_conclusion and len(content) > 2000:
                return True
        
        # Python files
        if target_file.endswith('.py'):
            # Check for balanced brackets/parentheses
            if content.count('(') != content.count(')'):
                return False
            if content.count('[') != content.count(']'):
                return False
            if content.count('{') != content.count('}'):
                return False
                
            # Check for complete structure (reasonable length + proper ending)
            lines = [line for line in content.split('\n') if line.strip()]
            if len(lines) > 20:  # Has reasonable content
                last_line = lines[-1].strip()
                # Should not end with incomplete statements
                if not (last_line.endswith(':') or 
                       last_line.endswith('\\') or
                       last_line.endswith(',')):
                    return True
        
        # JavaScript/TypeScript files
        if target_file.endswith(('.js', '.ts', '.jsx', '.tsx')):
            # Check for balanced brackets
            if content.count('{') != content.count('}'):
                return False
            if content.count('(') != content.count(')'):
                return False
                
            lines = [line for line in content.split('\n') if line.strip()]
            if len(lines) > 20:
                last_line = lines[-1].strip()
                # Complete if ends with proper syntax
                if (last_line.endswith('}') or 
                    last_line.endswith(';') or
                    last_line.endswith('*/') or
                    last_line.startswith('//')):
                    return True
        
        # 3. Generic checks for all file types
        if len(content) > 3000:  # Reasonable length
            # Check if it ends with complete sentences/sections
            lines = content.split('\n')
            last_lines = [line.strip() for line in lines[-10:] if line.strip()]
            
            if last_lines:
                last_line = last_lines[-1]
                # Complete if ends with proper punctuation or closing tags
                complete_endings = [
                    '.',      # Sentence
                    '```',    # Code block
                    '---',    # Section divider
                    '</details>',  # HTML details
                    '}',      # Closing brace
                    ';',      # Statement end
                    '*/',     # Comment end
                ]
                
                if any(last_line.endswith(ending) for ending in complete_endings):
                    return True
                    
        return False

    def _generate_continuation(self, target_file: str, evaluation_report: dict, 
                             context: str, existing_content: str) -> tuple[str, dict]:
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
            elif "suggestions" in evaluation_report and isinstance(evaluation_report["suggestions"], list):
                total_suggestions = len(evaluation_report["suggestions"])
        
        # Use the centralized continuation prompt template
        continuation_prompt = LLM_CONTINUATION_PROMPT.format(
            target_file=target_file,
            existing_content_tail=existing_content[-1000:],  # Last 1000 chars for context
            total_suggestions=total_suggestions,
            evaluation_report_excerpt=json.dumps(evaluation_report)[:4000],
            context_excerpt=context[:2000],
        )
        
        content, token_usage = conv.generate(
            system_prompt=continuation_prompt, 
            instruction_prompt="Continue the document from where it left off."
        )
        return content.strip(), token_usage

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
            context=context[:2500],
            guidance=(suggestion.content_guidance or "").strip(),
        )
        content, token_usage = conv.generate(system_prompt=system_prompt, instruction_prompt="Write the section content now.")
        return content.strip(), token_usage

    def generate_full_document(self, target_file: str, evaluation_report: dict, context: str = "", original_content: str = None) -> tuple[str, dict]:
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
                "max_tokens": getattr(llm, 'max_tokens', 16384)
            }
        }
        
        # Save debug info to file
        debug_dir = "outputs/debug_generation"
        os.makedirs(debug_dir, exist_ok=True)
        safe_filename = target_file.replace("/", "_").replace(".", "_")
        debug_file = os.path.join(debug_dir, f"{safe_filename}_debug.json")
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2, ensure_ascii=False)
        
        # Debug: Save raw evaluation_report to see what's being serialized
        eval_report_file = os.path.join(debug_dir, f"{safe_filename}_raw_eval_report.json")
        with open(eval_report_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        
        # Use comprehensive README prompt for README.md files
        if target_file.endswith("README.md"):
            system_prompt = LLM_README_COMPREHENSIVE_PROMPT.format(
                target_file=target_file,
                evaluation_report=json.dumps(evaluation_report)[:6000],
                context=context[:4000],
            )
        else:
            # Calculate total suggestions for the prompt
            total_suggestions = 1
            if isinstance(evaluation_report, dict):
                if "total_suggestions" in evaluation_report:
                    total_suggestions = evaluation_report["total_suggestions"]
                elif "suggestions" in evaluation_report and isinstance(evaluation_report["suggestions"], list):
                    total_suggestions = len(evaluation_report["suggestions"])
            
            system_prompt = LLM_FULLDOC_PROMPT.format(
                target_file=target_file,
                evaluation_report=json.dumps(evaluation_report)[:6000],
                context=context[:4000],
                total_suggestions=total_suggestions,
            )
        
        # Save initial prompt for debugging
        prompt_file = os.path.join(debug_dir, f"{safe_filename}_prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write("=== SYSTEM PROMPT ===\n")
            f.write(system_prompt)
            f.write("\n\n=== INSTRUCTION PROMPT ===\n")
            f.write("Write the full document now.")
            # Context is already embedded in system prompt; avoid duplicating here
        
        # Initial generation
        # If the original document is long (RMarkdown > 8k chars), avoid truncation by chunked rewrite
        # Lower threshold from 12k to 8k to catch more documents that would otherwise truncate
        use_chunked = bool(target_file.endswith('.Rmd') and isinstance(original_content, str) and len(original_content) > 8000)
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
            content, token_usage = conv.generate(system_prompt=system_prompt, instruction_prompt="Write the full document now.")
            content = content.strip()
        
        # Save initial generation for debugging
        generation_file = os.path.join(debug_dir, f"{safe_filename}_generation_0.txt")
        with open(generation_file, 'w', encoding='utf-8') as f:
            f.write(f"=== INITIAL GENERATION ===\n")
            f.write(f"Tokens: {token_usage}\n")
            f.write(f"Length: {len(content)} characters\n")
            if original_content:
                f.write(f"Original length: {len(original_content)} characters\n")
            f.write(f"Truncation detected: {self._detect_truncation(content, target_file, original_content)}\n")
            f.write(f"\n=== CONTENT ===\n")
            f.write(content)
        
        # Check for truncation and continue if needed
        max_continuations = 3  # Limit to prevent infinite loops
        continuation_count = 0
        
        while (not use_chunked and self._detect_truncation(content, target_file, original_content) and 
               continuation_count < max_continuations):
            
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
                elif "suggestions" in evaluation_report and isinstance(evaluation_report["suggestions"], list):
                    total_suggestions = len(evaluation_report["suggestions"])
            
            # Find better continuation point - look for last complete section
            continuation_point = self._find_continuation_point(content, original_content)
            if not continuation_point:
                continuation_point = content[-1000:]  # Fallback to last 1000 chars

            # Generate continuation prompt using centralized template
            continuation_prompt = LLM_CONTINUATION_PROMPT.format(
                target_file=target_file,
                existing_content_tail=continuation_point,
                total_suggestions=total_suggestions,
                evaluation_report_excerpt=json.dumps(evaluation_report)[:4000],
                context_excerpt=context[:2000],
            )
            
            # Save continuation prompt for debugging
            continuation_prompt_file = os.path.join(debug_dir, f"{safe_filename}_continuation_{continuation_count}_prompt.txt")
            with open(continuation_prompt_file, 'w', encoding='utf-8') as f:
                f.write(continuation_prompt)
            
            # Generate continuation
            continuation_content, continuation_usage = self._generate_continuation(
                target_file=target_file,
                evaluation_report=evaluation_report,
                context=context,
                existing_content=content
            )
            
            # Save continuation generation for debugging
            continuation_file = os.path.join(debug_dir, f"{safe_filename}_continuation_{continuation_count}.txt")
            with open(continuation_file, 'w', encoding='utf-8') as f:
                f.write(f"=== CONTINUATION {continuation_count} ===\n")
                f.write(f"Tokens: {continuation_usage}\n")
                f.write(f"Length: {len(continuation_content)} characters\n")
                f.write(f"Truncation detected: {self._detect_truncation(continuation_content, target_file)}\n")
                f.write(f"\n=== CONTENT ===\n")
                f.write(continuation_content)
            
            # Merge continuation with existing content
            if continuation_content:
                content += "\n\n" + continuation_content
                # Update token usage
                token_usage = {
                    "total_tokens": token_usage.get("total_tokens", 0) + continuation_usage.get("total_tokens", 0),
                    "prompt_tokens": token_usage.get("prompt_tokens", 0) + continuation_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0) + continuation_usage.get("completion_tokens", 0),
                }
                
                # Save merged content for debugging
                merged_file = os.path.join(debug_dir, f"{safe_filename}_merged_{continuation_count}.txt")
                with open(merged_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== MERGED CONTENT AFTER CONTINUATION {continuation_count} ===\n")
                    f.write(f"Total length: {len(content)} characters\n")
                    f.write(f"Truncation detected: {self._detect_truncation(content, target_file)}\n")
                    f.write(f"\n=== CONTENT ===\n")
                    f.write(content)
            else:
                # If continuation is empty, break to avoid infinite loop
                break
        
        # Clean up any markdown code fences that might have been added
        content = self._clean_markdown_fences(content)
        
        # Save final cleaned content for debugging
        final_file = os.path.join(debug_dir, f"{safe_filename}_final.txt")
        with open(final_file, 'w', encoding='utf-8') as f:
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
        if content.startswith('```markdown\n'):
            content = content[12:]  # Remove ```markdown\n
        
        # Remove ``` at the end
        if content.endswith('\n```'):
            content = content[:-4]  # Remove \n```
        elif content.endswith('```'):
            content = content[:-3]  # Remove ```
        
        # Remove any standalone ```markdown lines
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() == '```markdown':
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _split_rmd_into_chunks(self, content: str) -> list[dict]:
        chunks = []
        if not content:
            return chunks
        lines = content.split('\n')
        n = len(lines)
        i = 0
        if n >= 3 and lines[0].strip() == '---':
            j = 1
            while j < n and lines[j].strip() != '---':
                j += 1
            if j < n and lines[j].strip() == '---':
                chunks.append({"type": "yaml", "content": '\n'.join(lines[0:j+1])})
                i = j + 1
        buffer = []
        in_code = False
        for k in range(i, n):
            line = lines[k]
            if line.strip().startswith('```'):
                if in_code:
                    buffer.append(line)
                    chunks.append({"type": "code", "content": '\n'.join(buffer)})
                    buffer = []
                    in_code = False
                else:
                    if buffer and any(s.strip() for s in buffer):
                        chunks.append({"type": "text", "content": '\n'.join(buffer)})
                    buffer = [line]
                    in_code = True
            else:
                buffer.append(line)
        if buffer and any(s.strip() for s in buffer):
            chunks.append({"type": "code" if in_code else "text", "content": '\n'.join(buffer)})
        return chunks

    def _generate_text_chunk(self, conv: CommonConversation, evaluation_report: dict, context: str, chunk_text: str) -> tuple[str, dict]:
        LLM_CHUNK_PROMPT = (
            "You are BioGuider improving a single markdown chunk of a larger RMarkdown document.\n\n"
            "GOAL\nRefine ONLY the given chunk's prose per evaluation suggestions while preserving structure.\n"
            "Do not add conclusions or new sections.\n\n"
            "INPUTS\n- evaluation_report: <<{evaluation_report}>>\n- repo_context_excerpt: <<{context}>>\n- original_chunk:\n<<<\n{chunk}\n>>>\n\n"
            "RULES\n- Preserve headers/formatting in this chunk.\n- Do not invent technical specs.\n- Output ONLY the refined chunk (no fences)."
        )
        system_prompt = LLM_CHUNK_PROMPT.format(
            evaluation_report=json.dumps(evaluation_report)[:4000],
            context=context[:1500],
            chunk=chunk_text[:6000],
        )
        content, usage = conv.generate(system_prompt=system_prompt, instruction_prompt="Rewrite this chunk now.")
        return content.strip(), usage

    def _generate_full_document_chunked(self, target_file: str, evaluation_report: dict, context: str, original_content: str, debug_dir: str, safe_filename: str) -> tuple[str, dict]:
        conv = CommonConversation(self.llm)
        chunks = self._split_rmd_into_chunks(original_content)
        merged = []
        total_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        from datetime import datetime
        for idx, ch in enumerate(chunks):
            if ch["type"] in ("yaml", "code"):
                merged.append(ch["content"])
                continue
            out, usage = self._generate_text_chunk(conv, evaluation_report, context, ch["content"])
            if not out:
                out = ch["content"]
            merged.append(out)
            try:
                total_usage["total_tokens"] += int(usage.get("total_tokens", 0))
                total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                total_usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
            except Exception:
                pass
            chunk_file = os.path.join(debug_dir, f"{safe_filename}_chunk_{idx}.txt")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(f"=== CHUNK {idx} ({ch['type']}) at {datetime.now().isoformat()} ===\n")
                f.write(out)
        content = '\n'.join(merged)
        return content, total_usage


