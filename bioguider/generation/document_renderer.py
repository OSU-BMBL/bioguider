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

        # Other edit types (insert_after_header, replace_block) can be added as needed

        return content, {"added_lines": added}


