from __future__ import annotations

from typing import List, Dict

from .models import SuggestionItem, StyleProfile, DocumentPlan, PlannedEdit


class ChangePlanner:
    def build_plan(
        self,
        repo_path: str,
        style: StyleProfile,
        suggestions: List[SuggestionItem],
        available_files: Dict[str, str],
    ) -> DocumentPlan:
        planned: List[PlannedEdit] = []
        seen_headers: set[tuple[str, str]] = set()

        def section_header(title: str) -> str:
            # use heading level 2 for inserts to be safe
            h = style.heading_style or "#"
            return f"{h*2} {title}\n\n"

        for s in suggestions:
            for target in s.target_files:
                if target not in available_files:
                    # allow planning; renderer will skip if missing
                    pass

                if s.action == "add_dependencies_section":
                    content = section_header("Dependencies") + "- List required packages and versions.\n"
                    header_key = (target, (s.anchor_hint or "Dependencies").strip().lower())
                    if header_key in seen_headers:
                        continue
                    planned.append(PlannedEdit(
                        file_path=target,
                        edit_type="append_section",
                        anchor={"type": "header", "value": s.anchor_hint or "Dependencies"},
                        content_template=content,
                        rationale=s.source.get("evidence", ""),
                        suggestion_id=s.id,
                    ))
                    seen_headers.add(header_key)
                elif s.action == "add_system_requirements_section":
                    content = section_header("System Requirements") + "- OS and R version requirements.\n"
                    header_key = (target, (s.anchor_hint or "System Requirements").strip().lower())
                    if header_key in seen_headers:
                        continue
                    planned.append(PlannedEdit(
                        file_path=target,
                        edit_type="append_section",
                        anchor={"type": "header", "value": s.anchor_hint or "System Requirements"},
                        content_template=content,
                        rationale=s.source.get("evidence", ""),
                        suggestion_id=s.id,
                    ))
                    seen_headers.add(header_key)
                elif s.action == "mention_license_section":
                    content = section_header("License") + "This project is released under the MIT License. See LICENSE for details.\n"
                    header_key = (target, (s.anchor_hint or "License").strip().lower())
                    if header_key in seen_headers:
                        continue
                    planned.append(PlannedEdit(
                        file_path=target,
                        edit_type="append_section",
                        anchor={"type": "header", "value": s.anchor_hint or "License"},
                        content_template=content,
                        rationale=s.source.get("evidence", ""),
                        suggestion_id=s.id,
                    ))
                    seen_headers.add(header_key)
                elif s.action == "normalize_headings_structure":
                    # Minimal placeholder: avoid heavy rewrites
                    # Plan a no-op or a small note; actual normalization could be added later
                    continue
                elif s.action == "add_usage_section":
                    content = section_header("Usage") + "- Brief example of typical workflow.\n"
                    header_key = (target, "usage")
                    if header_key in seen_headers:
                        continue
                    planned.append(PlannedEdit(
                        file_path=target,
                        edit_type="append_section",
                        anchor={"type": "header", "value": "Usage"},
                        content_template=content,
                        rationale=s.source.get("evidence", ""),
                        suggestion_id=s.id,
                    ))
                    seen_headers.add(header_key)
                elif s.action == "replace_intro":
                    # Replace intro block (between H1 and first H2) with a clean Overview section
                    content = section_header("Overview") + "- Clear 2–3 sentence summary of purpose and audience.\n"
                    header_key = (target, "overview")
                    if header_key in seen_headers:
                        continue
                    planned.append(PlannedEdit(
                        file_path=target,
                        edit_type="replace_intro_block",
                        anchor={"type": "header", "value": "Overview"},
                        content_template=content,
                        rationale=s.source.get("evidence", ""),
                        suggestion_id=s.id,
                    ))
                    seen_headers.add(header_key)
                elif s.action == "clarify_mandatory_vs_optional":
                    content = section_header("Dependencies") + (
                        "- Mandatory: ...\n- Optional: ...\n"
                    )
                    header_key = (target, "dependencies")
                    if header_key in seen_headers:
                        continue
                    planned.append(PlannedEdit(
                        file_path=target,
                        edit_type="append_section",
                        anchor={"type": "header", "value": "Dependencies"},
                        content_template=content,
                        rationale=s.source.get("evidence", ""),
                        suggestion_id=s.id,
                    ))
                    seen_headers.add(header_key)
                elif s.action == "add_hardware_requirements":
                    content = section_header("Hardware Requirements") + (
                        "- Recommended: >=16 GB RAM, multi-core CPU for large datasets.\n"
                    )
                    header_key = (target, (s.anchor_hint or "Hardware Requirements").strip().lower())
                    if header_key in seen_headers:
                        continue
                    planned.append(PlannedEdit(
                        file_path=target,
                        edit_type="append_section",
                        anchor={"type": "header", "value": s.anchor_hint or "Hardware Requirements"},
                        content_template=content,
                        rationale=s.source.get("evidence", ""),
                        suggestion_id=s.id,
                    ))
                    seen_headers.add(header_key)

        return DocumentPlan(repo_path=repo_path, style_profile=style, planned_edits=planned)


