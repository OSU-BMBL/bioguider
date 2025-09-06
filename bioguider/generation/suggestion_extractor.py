from __future__ import annotations

from typing import List
from .models import EvaluationReport, SuggestionItem


class SuggestionExtractor:
    def extract(self, report: EvaluationReport) -> List[SuggestionItem]:
        suggestions: List[SuggestionItem] = []

        # README-related suggestions
        if report.readme_evaluation:
            for file_name, evaluation in report.readme_evaluation.items():
                structured = evaluation.get("structured_evaluation") if isinstance(evaluation, dict) else None
                if structured:
                    # Intro cleanup / overview enhancement beyond explicit suggestions
                    suggestions.append(SuggestionItem(
                        id=f"readme-intro-cleanup-{file_name}",
                        category="readme.intro_cleanup",
                        severity="should_fix",
                        source={"section": "readme", "field": "overview", "evidence": "Improve top-level overview for clarity and tone."},
                        target_files=[file_name],
                        action="replace_intro",
                        anchor_hint="Overview",
                        content_guidance="Rewrite the opening summary to be clear, neutral, and typo-free.",
                    ))
                    # Dependency clarity
                    dep_score = structured.get("dependency_score")
                    dep_sugg = structured.get("dependency_suggestions")
                    if dep_score in ("Poor", "Fair") or dep_sugg:
                        suggestions.append(SuggestionItem(
                            id=f"readme-dependencies-{file_name}",
                            category="readme.dependencies",
                            severity="should_fix",
                            source={"section": "readme", "field": "dependencies", "evidence": str(dep_sugg or dep_score)},
                            target_files=[file_name],
                            action="add_dependencies_section",
                            anchor_hint="Dependencies",
                            content_guidance=str(dep_sugg or ""),
                        ))

                    # Hardware/Software specs
                    hw_score = structured.get("hardware_and_software_spec_score")
                    hw_sugg = structured.get("hardware_and_software_spec_suggestions")
                    if hw_score in ("Poor", "Fair") or hw_sugg:
                        suggestions.append(SuggestionItem(
                            id=f"readme-sysreq-{file_name}",
                            category="readme.system_requirements",
                            severity="should_fix",
                            source={"section": "readme", "field": "hardware_and_software", "evidence": str(hw_sugg or hw_score)},
                            target_files=[file_name],
                            action="add_system_requirements_section",
                            anchor_hint="System Requirements",
                            content_guidance=str(hw_sugg or ""),
                        ))

                    # License mention
                    lic_sugg = structured.get("license_suggestions")
                    lic_score = structured.get("license_score")
                    if lic_sugg and lic_score:
                        suggestions.append(SuggestionItem(
                            id=f"readme-license-{file_name}",
                            category="readme.license",
                            severity="nice_to_have",
                            source={"section": "readme", "field": "license", "evidence": str(lic_sugg)},
                            target_files=[file_name],
                            action="mention_license_section",
                            anchor_hint="License",
                            content_guidance=str(lic_sugg),
                        ))

                    # Readability structuring
                    read_sugg = structured.get("readability_suggestions")
                    if read_sugg:
                        suggestions.append(SuggestionItem(
                            id=f"readme-structure-{file_name}",
                            category="readme.readability",
                            severity="nice_to_have",
                            source={"section": "readability", "field": "readability_suggestions", "evidence": str(read_sugg)},
                            target_files=[file_name],
                            action="normalize_headings_structure",
                            anchor_hint="Installation",
                            content_guidance=str(read_sugg),
                        ))
                        # If suggestions mention Usage, add a usage section
                        if isinstance(read_sugg, str) and "Usage" in read_sugg:
                            suggestions.append(SuggestionItem(
                                id=f"readme-usage-{file_name}",
                                category="readme.usage",
                                severity="nice_to_have",
                                source={"section": "readability", "field": "usage", "evidence": "Add Usage section as suggested."},
                                target_files=[file_name],
                                action="add_usage_section",
                                anchor_hint="Usage",
                                content_guidance="Provide a brief usage example and key commands.",
                            ))

        # Installation-related suggestions
        if report.installation_evaluation:
            structured = None
            if isinstance(report.installation_evaluation, dict):
                structured = report.installation_evaluation.get("structured_evaluation")
            if structured:
                dep_sugg = structured.get("dependency_suggestions")
                if dep_sugg:
                    for target in report.installation_files or []:
                        suggestions.append(SuggestionItem(
                            id=f"install-dep-clarify-{target}",
                            category="installation.dependencies",
                            severity="should_fix",
                            source={"section": "installation", "field": "dependency_suggestions", "evidence": str(dep_sugg)},
                            target_files=[target],
                            action="clarify_mandatory_vs_optional",
                            anchor_hint="Dependencies",
                            content_guidance=str(dep_sugg),
                        ))
                hw = structured.get("hardware_requirements")
                if hw is False:
                    for target in report.installation_files or []:
                        suggestions.append(SuggestionItem(
                            id=f"install-hw-req-{target}",
                            category="installation.hardware",
                            severity="should_fix",
                            source={"section": "installation", "field": "hardware_requirements", "evidence": "not specified"},
                            target_files=[target],
                            action="add_hardware_requirements",
                            anchor_hint="Hardware Requirements",
                            content_guidance="Add concise RAM/CPU recommendation as per report guidance.",
                        ))

        # Submission requirements could drive expected output/dataset sections; use only if in files list
        # Keep minimal to avoid speculative content

        return suggestions


