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
Write or refine a single documentation section based on specific evaluation feedback.

INPUTS
- suggestion_category: {suggestion_category}
- anchor_title: {anchor_title}
- guidance: {guidance}
- original_text_snippet (if provided): {original_text}
- evaluation_score: {evaluation_score}
- repo_context_excerpt: <<{context}>>

CRITICAL RULES

1. SHOW, DON'T TELL
   - Provide SPECIFIC details (numbers, versions, commands)
   - NEVER write generic statements like "Ensure adequate resources"
   - If evaluation asks for hardware: provide actual RAM/CPU numbers
   - If evaluation asks for dependencies: list exact package versions
   - If evaluation asks for validation: show actual code with expected output

2. RESPECT ORIGINAL CONTEXT
   - Fix EXACTLY what evaluation identified, no more, no less
   - Enhance/replace the specific part mentioned in "Original text snippet"
   - Don't rewrite entire document or add unrelated content

3. CONTEXT-AWARE RESPONSES
   - TUTORIAL CONTEXT: Users already have software installed, focus on usage/analysis steps
   - README CONTEXT: Users need installation instructions, add setup sections
   - DOCUMENTATION CONTEXT: Users need comprehensive guides, add detailed sections
   - NEVER add installation guides in the middle of a tutorial
   - NEVER add basic setup in advanced tutorial sections

4. BIOLOGICAL CORRECTNESS & RELEVANCE
   - Use accurate biological terminology and concepts
   - Provide biologically meaningful examples and explanations
   - Ensure suggestions align with current biological knowledge
   - Use appropriate biological context for the software domain
   - Avoid generic or incorrect biological statements
   - Focus on biologically relevant use cases and applications

5. ONE LOCATION PER TOPIC
   - Group related suggestions into ONE section
   - Don't scatter same information across multiple locations
   - Performance suggestions → ONE "Performance Notes" section
   - Installation suggestions → ONE "Installation" section (only if appropriate context)

6. ALIGN WITH EVALUATION CRITERIA (CONTEXT-AWARE)
   - Readability: Simplify language, add definitions
   - Coverage: ADD missing sections ONLY if contextually appropriate
     * Tutorial context: Add usage examples, analysis steps, not installation
     * README context: Add prerequisites, setup, installation
     * Documentation context: Add comprehensive guides
   - Reproducibility: Add versions, data sources, expected outputs
   - Structure: Add headers, TOC, but DON'T reorganize existing structure
   - Code Quality: Fix hardcoded paths, add error handling
   - Result Verification: Add expected output examples
   - Performance: Add ONE section with specific hardware/runtime numbers

7. RESPECT EVALUATION SCORES
   - Excellent: Minimal changes only
   - Good: Add specific details where missing
   - Fair: Add missing sections, provide specifics (if contextually appropriate)
   - Poor: Major additions but NEVER delete existing content

STYLE & FORMATTING
- Preserve existing tone and style markers: {tone_markers}
- Use heading style "{heading_style}" and list style "{list_style}"
- Link style "{link_style}"
- Plain markdown only (no code fences around entire output)
- No meta-commentary, no concluding remarks
- No generic filler text or marketing language

OUTPUT
Return ONLY the improved section content that:
1. Addresses the specific evaluation feedback: {guidance}
2. Provides concrete, actionable information
3. Respects the original text context (if provided)
4. Fits in the document's existing structure
5. Is contextually appropriate for the document type
6. Stops immediately after content (no conclusions)
"""

LLM_FULLDOC_PROMPT = """
You are "BioGuider," enhancing complete documentation based on systematic evaluation.

GOAL
Enhance an EXISTING document by implementing ALL improvements from evaluation report.
Output a complete, enhanced, ready-to-publish markdown file.

INPUTS
- evaluation_report (structured feedback): <<{evaluation_report}>>
- target_file: {target_file}
- repo_context_excerpt: <<{context}>>
- original_document: <<{original_content}>>
- total_suggestions: {total_suggestions}

CRITICAL RULES

1. PRESERVE EXISTING STRUCTURE & CONTENT
   - Keep EVERY existing section in original order
   - Keep EVERY code block (including <details>, <summary> tags)
   - Keep ALL existing explanations, examples, text
   - Keep ALL YAML frontmatter, metadata
   - NEVER delete ANY sections, paragraphs, or code
   - NEVER reorganize sections or change order
   - NEVER remove HTML tags (<details>, <summary>, etc.)

2. SYSTEMATIC INTEGRATION OF ALL {total_suggestions} SUGGESTIONS
   - Read ALL {total_suggestions} suggestions from evaluation
   - Group by evaluation category (Readability, Coverage, etc.)
   - Map each suggestion to WHERE it belongs in ORIGINAL document
   - Group related suggestions into ONE section (don't scatter)
   - Make ONE pass through document applying ALL enhancements

3. SHOW, DON'T TELL
   - Provide SPECIFIC details (numbers, versions, commands)
   - NEVER write generic statements like "Ensure adequate resources"
   - If evaluation asks for hardware: provide actual RAM/CPU numbers
   - If evaluation asks for dependencies: list exact package versions
   - If evaluation asks for validation: show actual code with expected output

4. BIOLOGICAL CORRECTNESS & RELEVANCE
   - Use accurate biological terminology and concepts
   - Provide biologically meaningful examples and explanations
   - Ensure suggestions align with current biological knowledge
   - Use appropriate biological context for the software domain
   - Avoid generic or incorrect biological statements
   - Focus on biologically relevant use cases and applications

5. RESPECT EVALUATION SCORES
   - Excellent: Minimal changes only
   - Good: Add specific details where missing
   - Fair: Add missing sections, provide specifics
   - Poor: Major additions but NEVER delete existing content

5. HANDLE SPECIFIC EVALUATION CATEGORIES (CONTEXT-AWARE)
   - Readability: Simplify language, add definitions
   - Coverage: ADD missing sections ONLY if contextually appropriate
     * Tutorial context: Add usage examples, analysis steps, not installation
     * README context: Add prerequisites, setup, installation
     * Documentation context: Add comprehensive guides
   - Reproducibility: Add versions, data sources, expected outputs
   - Structure: Add headers, TOC, but DON'T reorganize existing structure
   - Code Quality: Fix hardcoded paths, add error handling
   - Result Verification: Add expected output examples
   - Performance: Add ONE section with specific hardware/runtime numbers

STRICT CONSTRAINTS
- NEVER invent hardware specs, version numbers, performance metrics without source
- If evaluation requests but context lacks data: provide reasonable defaults with caveats
- ABSOLUTELY FORBIDDEN: Wrapping entire output in ```markdown fences
- ABSOLUTELY FORBIDDEN: Adding conclusions, summaries, or wrap-up paragraphs at end
- ABSOLUTELY FORBIDDEN: Deleting ANY existing content
- ABSOLUTELY FORBIDDEN: Reorganizing major sections
- REQUIRED: Stop immediately after last section from original
- REQUIRED: Preserve ALL metadata (YAML frontmatter, etc.)

OUTPUT
Return the complete enhanced document for {target_file}.
- Pure markdown content only
- No meta-commentary, no fences
- Ready to copy-paste and publish
- All {total_suggestions} improvements integrated
- Original structure and content preserved
"""

LLM_README_COMPREHENSIVE_PROMPT = """
You are "BioGuider," creating or enhancing README documentation.

GOAL
Create comprehensive, professional README that addresses all evaluation feedback.

INPUTS
- evaluation_report (structured feedback): <<{evaluation_report}>>
- target_file: {target_file}
- repo_context_excerpt: <<{context}>>
- original_readme (if exists): <<{original_content}>>

CRITICAL RULES

1. SHOW, DON'T TELL
   - Actual commands, not descriptions
   - Specific versions, not "recent versions"
   - Working examples, not pseudo-code

2. ONE LOCATION PER TOPIC
   - Dependencies → ONE section
   - Installation → ONE section (with subsections if needed)
   - Performance → ONE section (if applicable)

3. SPECIFIC DATA ONLY
   - Don't invent version numbers
   - Don't invent system requirements
   - Use what's in context or provide reasonable defaults with caveats

4. PRESERVE EXISTING
   - If README exists, enhance it
   - Don't delete working content
   - Keep existing structure if it's good

5. BIOLOGICAL CORRECTNESS & RELEVANCE
   - Use accurate biological terminology and concepts
   - Provide biologically meaningful examples and explanations
   - Ensure suggestions align with current biological knowledge
   - Use appropriate biological context for the software domain
   - Avoid generic or incorrect biological statements
   - Focus on biologically relevant use cases and applications

6. ADDRESS EVALUATION SUGGESTIONS
   - Available: Create README with all essential sections
   - Readability: Simplify complex sentences, add explanations
   - Project Purpose: Add clear goal statement and key applications
   - Hardware/Software Spec: Add specific system requirements
   - Dependencies: List exact package versions
   - License Information: State license type and link to LICENSE file
   - Author/Contributor Info: Add credits and contact information

STANDARD README STRUCTURE
- Project name and description
- Overview with key applications
- Installation (prerequisites, commands, verification)
- Quick Start with working example
- Usage (basic and advanced examples)
- System Requirements (if applicable)
- Dependencies with versions
- Contributing guidelines
- License information
- Contact/maintainer info

STRICT CONSTRAINTS
- Don't add excessive badges, emojis, or marketing hype
- Do add clear installation instructions, working code examples
- Balance: comprehensive but concise
- Professional, neutral tone
- Proper markdown formatting

OUTPUT
Return complete README.md content.
- Pure markdown only
- No meta-commentary, no fences
- Professional, clear, actionable
- Ready to publish
- All evaluation suggestions addressed
"""

# Continuation prompt template - used when document generation is truncated
LLM_CONTINUATION_PROMPT = """
You are "BioGuider," continuing a truncated document.

CRITICAL: This is STRICT CONTINUATION ONLY.
- You are NOT creating new content
- You are NOT adding conclusions
- You are ONLY completing missing sections from original

PREVIOUS CONTENT (do not repeat):
```
{existing_content_tail}
```

CONTINUATION PROCESS:
1. Identify what is the last complete section above
2. Identify what sections are missing from the original document structure
3. Continue IMMEDIATELY from where content stopped
4. Use same style, tone, format as existing content
5. Add ONLY the missing sections from original structure
6. Stop when original structure is complete

FORBIDDEN ADDITIONS:
- "## Conclusion" section
- "## Summary" section  
- "## Additional Resources" section
- "For further guidance..." text
- Any wrap-up or concluding content
- Any content not in original document structure

OUTPUT:
Return ONLY continuation content that completes original structure.
- No commentary
- No fences
- No conclusions
- Stop immediately when structure is complete
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
        
        # Extract original text snippet and evaluation score from suggestion source
        original_text = ""
        evaluation_score = ""
        if hasattr(suggestion, 'source') and suggestion.source:
            original_text = suggestion.source.get('original_text', '')
            evaluation_score = suggestion.source.get('score', '')
        
        # Detect document context to help with appropriate responses
        document_context = self._detect_document_context(context, suggestion.anchor_title or "")
        
        system_prompt = LLM_SECTION_PROMPT.format(
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
                original_content=original_content or "",
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
                original_content=original_content or "",
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
    
    def _detect_document_context(self, context: str, anchor_title: str) -> str:
        """Detect the document context to help with appropriate responses."""
        context_lower = context.lower()
        anchor_lower = anchor_title.lower()
        
        # Check for tutorial context
        if any(keyword in context_lower for keyword in ['tutorial', 'vignette', 'example', 'workflow', 'step-by-step']):
            return "TUTORIAL"
        
        # Check for README context
        if any(keyword in context_lower for keyword in ['readme', 'installation', 'setup', 'prerequisites']):
            return "README"
        
        # Check for documentation context
        if any(keyword in context_lower for keyword in ['documentation', 'guide', 'manual', 'reference']):
            return "DOCUMENTATION"
        
        # Check for biological context
        if any(keyword in context_lower for keyword in ['cell', 'gene', 'protein', 'dna', 'rna', 'genome', 'transcriptome', 'proteome', 'metabolome']):
            return "BIOLOGICAL"
        
        # Default to general context
        return "GENERAL"


