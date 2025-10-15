from __future__ import annotations

from typing import Tuple

from .models import PlannedEdit


class DocumentRenderer:
    def apply_edit(self, original: str, edit: PlannedEdit) -> Tuple[str, dict]:
        content = original
        added = 0

        if edit.edit_type == "append_section":
            # Avoid duplicate header if the same header already exists
            header_line = None
            if edit.content_template.lstrip().startswith("#"):
                header_line = edit.content_template.strip().splitlines()[0].strip()
            if header_line and header_line in content:
                return content, {"added_lines": 0}
            # Append with two leading newlines if needed
            sep = "\n\n" if not content.endswith("\n\n") else ""
            content = f"{content}{sep}{edit.content_template}"
            added = len(edit.content_template.splitlines())

        elif edit.edit_type == "replace_intro_block":
            # Replace content from start to first level-2 header (##) with new intro
            lines = content.splitlines()
            end_idx = None
            for i, ln in enumerate(lines):
                if ln.strip().startswith("## "):
                    end_idx = i
                    break
            if end_idx is None:
                # No H2 header found; replace entire content
                new_content = edit.content_template
            else:
                head = lines[:0]
                tail = lines[end_idx:]
                new_content = edit.content_template.rstrip() + "\n\n" + "\n".join(tail)
            added = len(edit.content_template.splitlines())
            content = new_content

        elif edit.edit_type == "insert_after_header":
            # Insert content after a specific header, but integrate naturally
            header_value = edit.anchor.get("value", "")
            if header_value:
                lines = content.splitlines()
                insert_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith("#") and header_value.lower() in line.lower():
                        # Find a good insertion point after the header and its immediate content
                        insert_idx = i + 1
                        # Skip empty lines and find the first substantial content
                        while insert_idx < len(lines) and lines[insert_idx].strip() == "":
                            insert_idx += 1
                        # Insert after the first code block or paragraph, but before next major section
                        while insert_idx < len(lines):
                            line_content = lines[insert_idx].strip()
                            if line_content.startswith("#") and not line_content.startswith("###"):
                                break
                            if line_content.startswith("```") and insert_idx > 0:
                                # Found end of code block, insert after it
                                insert_idx += 1
                                break
                            insert_idx += 1
                        break
                
                if insert_idx is not None:
                    # Insert the new content with minimal formatting
                    new_content_lines = edit.content_template.splitlines()
                    # Remove standalone headers to avoid creating new major sections
                    filtered_lines = []
                    for line in new_content_lines:
                        if line.strip().startswith("## ") and len(line.strip()) < 50:
                            # Convert major headers to minor explanations
                            header_text = line.strip()[3:].strip()
                            filtered_lines.append(f"\n**Note:** {header_text.lower()}")
                        else:
                            filtered_lines.append(line)
                    
                    # Insert with minimal spacing
                    new_lines = lines[:insert_idx] + [""] + filtered_lines + lines[insert_idx:]
                    content = "\n".join(new_lines)
                    added = len(filtered_lines)
                else:
                    # Header not found, append at end
                    sep = "\n\n" if not content.endswith("\n\n") else ""
                    content = f"{content}{sep}{edit.content_template}"
                    added = len(edit.content_template.splitlines())
            else:
                # No header specified, append at end
                sep = "\n\n" if not content.endswith("\n\n") else ""
                content = f"{content}{sep}{edit.content_template}"
                added = len(edit.content_template.splitlines())

        elif edit.edit_type == "rmarkdown_integration":
            # Special handling for RMarkdown files - integrate content naturally
            header_value = edit.anchor.get("value", "")
            if header_value:
                lines = content.splitlines()
                insert_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith("#") and header_value.lower() in line.lower():
                        # Find insertion point after the first code block in this section
                        insert_idx = i + 1
                        while insert_idx < len(lines):
                            line_content = lines[insert_idx].strip()
                            if line_content.startswith("```") and insert_idx > 0:
                                # Found code block, insert after it
                                insert_idx += 1
                                break
                            if line_content.startswith("#") and not line_content.startswith("###"):
                                # Next major section, insert before it
                                break
                            insert_idx += 1
                        break
                
                if insert_idx is not None:
                    # Process content to be more contextual
                    new_content_lines = edit.content_template.splitlines()
                    contextual_lines = []
                    
                    for line in new_content_lines:
                        # Convert standalone sections to contextual notes
                        if line.strip().startswith("## "):
                            header_text = line.strip()[3:].strip()
                            contextual_lines.append(f"\n**Note:** For this tutorial, {header_text.lower()}")
                        elif line.strip().startswith("# "):
                            header_text = line.strip()[2:].strip()
                            contextual_lines.append(f"\n**Important:** {header_text.lower()}")
                        else:
                            contextual_lines.append(line)
                    
                    # Insert with minimal disruption
                    new_lines = lines[:insert_idx] + [""] + contextual_lines + lines[insert_idx:]
                    content = "\n".join(new_lines)
                    added = len(contextual_lines)
                else:
                    # Fallback to append
                    sep = "\n\n" if not content.endswith("\n\n") else ""
                    content = f"{content}{sep}{edit.content_template}"
                    added = len(edit.content_template.splitlines())
            else:
                sep = "\n\n" if not content.endswith("\n\n") else ""
                content = f"{content}{sep}{edit.content_template}"
                added = len(edit.content_template.splitlines())

        elif edit.edit_type == "full_replace":
            # Replace entire document content
            content = edit.content_template
            added = len(edit.content_template.splitlines())

        # Other edit types (replace_block) can be added as needed

        return content, {"added_lines": added}


