from __future__ import annotations

from typing import List
from .models import EvaluationReport, SuggestionItem


class SuggestionExtractor:
    def extract(self, report: EvaluationReport) -> List[SuggestionItem]:
        suggestions: List[SuggestionItem] = []

        # README-related suggestions - Extract specific suggestions
        if report.readme_evaluation:
            for file_name, evaluation in report.readme_evaluation.items():
                structured = evaluation.get("structured_evaluation") if isinstance(evaluation, dict) else None
                if structured:
                    # Extract specific dependency suggestions
                    dep_score = structured.get("dependency_score")
                    dep_suggestions = structured.get("dependency_suggestions")
                    if dep_score in ("Poor", "Fair") and dep_suggestions:
                        suggestions.append(SuggestionItem(
                            id=f"readme-dependencies-{file_name}",
                            category="readme.dependencies",
                            severity="should_fix",
                            source={"section": "readme", "field": "dependency_suggestions", "score": dep_score},
                            target_files=[file_name],
                            action="add_dependencies_section",
                            anchor_hint="Dependencies",
                            content_guidance=dep_suggestions,
                        ))
                    
                    # Extract specific hardware/software suggestions
                    hw_sw_score = structured.get("hardware_and_software_spec_score")
                    hw_sw_suggestions = structured.get("hardware_and_software_spec_suggestions")
                    if hw_sw_score in ("Poor", "Fair") and hw_sw_suggestions:
                        suggestions.append(SuggestionItem(
                            id=f"readme-hardware-{file_name}",
                            category="readme.hardware",
                            severity="should_fix",
                            source={"section": "readme", "field": "hardware_and_software_spec_suggestions", "score": hw_sw_score},
                            target_files=[file_name],
                            action="add_system_requirements_section",
                            anchor_hint="System Requirements",
                            content_guidance=hw_sw_suggestions,
                        ))
                    
                    # Extract specific project purpose suggestions
                    purpose_score = structured.get("project_purpose_score")
                    purpose_suggestions = structured.get("project_purpose_suggestions")
                    if purpose_score in ("Poor", "Fair") and purpose_suggestions:
                        suggestions.append(SuggestionItem(
                            id=f"readme-purpose-{file_name}",
                            category="readme.purpose",
                            severity="should_fix",
                            source={"section": "readme", "field": "project_purpose_suggestions", "score": purpose_score},
                            target_files=[file_name],
                            action="full_replace",
                            anchor_hint="Overview",
                            content_guidance=purpose_suggestions,
                        ))
                    
                    # Extract specific readability suggestions
                    readability_score = structured.get("readability_score")
                    readability_suggestions = structured.get("readability_suggestions")
                    if readability_score in ("Poor", "Fair") and readability_suggestions:
                        suggestions.append(SuggestionItem(
                            id=f"readme-readability-{file_name}",
                            category="readme.readability",
                            severity="should_fix",
                            source={"section": "readme", "field": "readability_suggestions", "score": readability_score},
                            target_files=[file_name],
                            action="full_replace",
                            anchor_hint="Introduction",
                            content_guidance=readability_suggestions,
                        ))

                    # Intro cleanup / overview enhancement beyond explicit suggestions
                    suggestions.append(SuggestionItem(
                        id=f"readme-intro-cleanup-{file_name}",
                        category="readme.intro_cleanup",
                        severity="should_fix",
                        source={"section": "readme", "field": "overview", "score": "Fair"},
                        target_files=[file_name],
                        action="replace_intro",
                        anchor_hint="Overview",
                        content_guidance="Rewrite the opening summary to be clear, neutral, and typo-free.",
                    ))
                    # Dependency clarity - prioritize specific suggestions (avoid duplicates)
                    dep_score = structured.get("dependency_score")
                    dep_sugg = structured.get("dependency_suggestions")
                    if dep_sugg and dep_score in ("Poor", "Fair"):  # Only if not already added above
                        suggestions.append(SuggestionItem(
                            id=f"readme-dependencies-clarify-{file_name}",
                            category="readme.dependencies",
                            severity="should_fix",
                            source={"section": "readme", "field": "dependencies", "score": dep_score},
                            target_files=[file_name],
                            action="add_dependencies_section",
                            anchor_hint="Dependencies",
                            content_guidance=str(dep_sugg),
                        ))
                    elif dep_score in ("Poor", "Fair"):  # Fallback to score-based approach
                        suggestions.append(SuggestionItem(
                            id=f"readme-dependencies-fallback-{file_name}",
                            category="readme.dependencies",
                            severity="should_fix",
                            source={"section": "readme", "field": "dependencies", "score": dep_score},
                            target_files=[file_name],
                            action="add_dependencies_section",
                            anchor_hint="Dependencies",
                            content_guidance="List R library dependencies and provide installation guide.",
                        ))

                    # Hardware/Software specs - prioritize specific suggestions (avoid duplicates)
                    hw_score = structured.get("hardware_and_software_spec_score")
                    hw_sugg = structured.get("hardware_and_software_spec_suggestions")
                    if hw_sugg and hw_score in ("Poor", "Fair"):  # Only if not already added above
                        suggestions.append(SuggestionItem(
                            id=f"readme-sysreq-clarify-{file_name}",
                            category="readme.system_requirements",
                            severity="should_fix",
                            source={"section": "readme", "field": "hardware_and_software", "score": hw_score},
                            target_files=[file_name],
                            action="add_system_requirements_section",
                            anchor_hint="System Requirements",
                            content_guidance=str(hw_sugg),
                        ))
                    elif hw_score in ("Poor", "Fair"):  # Fallback to score-based approach
                        suggestions.append(SuggestionItem(
                            id=f"readme-sysreq-fallback-{file_name}",
                            category="readme.system_requirements",
                            severity="should_fix",
                            source={"section": "readme", "field": "hardware_and_software", "score": hw_score},
                            target_files=[file_name],
                            action="add_system_requirements_section",
                            anchor_hint="System Requirements",
                            content_guidance="Specify R version requirements, recommend RAM/CPU configurations, and tailor installation instructions for platforms.",
                        ))

                    # License mention
                    lic_sugg = structured.get("license_suggestions")
                    lic_score = structured.get("license_score")
                    if lic_sugg and lic_score:
                        suggestions.append(SuggestionItem(
                            id=f"readme-license-{file_name}",
                            category="readme.license",
                            severity="nice_to_have",
                            source={"section": "readme", "field": "license"},
                            target_files=[file_name],
                            action="mention_license_section",
                            anchor_hint="License",
                            content_guidance=str(lic_sugg),
                        ))

                    # Readability structuring - prioritize specific suggestions (avoid duplicates)
                    read_sugg = structured.get("readability_suggestions")
                    read_score = structured.get("readability_score")
                    if read_sugg and read_score in ("Poor", "Fair"):  # Only if not already added above
                        suggestions.append(SuggestionItem(
                            id=f"readme-structure-clarify-{file_name}",
                            category="readme.readability",
                            severity="should_fix",
                            source={"section": "readability", "field": "readability_suggestions", "score": read_score},
                            target_files=[file_name],
                            action="normalize_headings_structure",
                            anchor_hint="Installation",
                            content_guidance=str(read_sugg),
                        ))
                    elif read_score in ("Poor", "Fair"):  # Fallback to score-based approach
                        suggestions.append(SuggestionItem(
                            id=f"readme-structure-fallback-{file_name}",
                            category="readme.readability",
                            severity="should_fix",
                            source={"section": "readability", "field": "readability_score", "score": read_score},
                            target_files=[file_name],
                            action="normalize_headings_structure",
                            anchor_hint="Installation",
                            content_guidance="Improve readability with better structure and formatting.",
                        ))
                        # If suggestions mention Usage, add a usage section
                        if isinstance(read_sugg, str) and "Usage" in read_sugg:
                            suggestions.append(SuggestionItem(
                                id=f"readme-usage-{file_name}",
                                category="readme.usage",
                                severity="nice_to_have",
                                source={"section": "readability", "field": "usage"},
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
                # Use full_replace mode for all installation files
                dep_sugg = structured.get("dependency_suggestions")
                hw_req = structured.get("hardware_requirements")
                compat_os = structured.get("compatible_os")
                overall = structured.get("overall_score")
                
                # Trigger full_replace for all installation files when needed
                if overall in ("Poor", "Fair") or hw_req is False or compat_os is False or dep_sugg:
                    for target in report.installation_files or []:
                        suggestions.append(SuggestionItem(
                            id=f"install-full-replace-{target}",
                            category="installation.full_replace",
                            severity="should_fix",
                            source={"section": "installation", "field": "overall"},
                            target_files=[target],
                            action="full_replace",
                            anchor_hint=None,
                            content_guidance="Comprehensive rewrite preserving original structure while adding improved dependencies, hardware requirements, and installation instructions.",
                        ))

        # Submission requirements could drive expected output/dataset sections; use only if in files list
        # Keep minimal to avoid speculative content

        # Userguide/API docs suggestions (new interface) - Extract specific suggestions
        if getattr(report, "userguide_evaluation", None) and isinstance(report.userguide_evaluation, dict):
            for file_name, eval_block in report.userguide_evaluation.items():
                ug_eval = eval_block.get("user_guide_evaluation") if isinstance(eval_block, dict) else None
                consistency_eval = eval_block.get("consistency_evaluation") if isinstance(eval_block, dict) else None
                
                if isinstance(ug_eval, dict):
                    # Extract specific readability suggestions
                    readability_score = ug_eval.get("readability_score", "")
                    readability_suggestions = ug_eval.get("readability_suggestions", [])
                    if readability_score in ("Poor", "Fair") and readability_suggestions:
                        for i, suggestion in enumerate(readability_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"userguide-readability-{file_name}-{i}",
                                    category="userguide.readability",
                                    severity="should_fix",
                                    source={"section": "userguide", "field": "readability_suggestions", "score": readability_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint=f"Readability-{i+1}",
                                    content_guidance=suggestion,
                                ))
                    
                    # Extract specific context and purpose suggestions
                    context_score = ug_eval.get("context_and_purpose_score", "")
                    context_suggestions = ug_eval.get("context_and_purpose_suggestions", [])
                    if context_score in ("Poor", "Fair") and context_suggestions:
                        for i, suggestion in enumerate(context_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"userguide-context-{file_name}-{i}",
                                    category="userguide.context",
                                    severity="should_fix",
                                    source={"section": "userguide", "field": "context_and_purpose_suggestions", "score": context_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint=f"Context-{i+1}",
                                    content_guidance=suggestion,
                                ))
                    
                    # Extract specific error handling suggestions
                    error_score = ug_eval.get("error_handling_score", "")
                    error_suggestions = ug_eval.get("error_handling_suggestions", [])
                    if error_score in ("Poor", "Fair") and error_suggestions:
                        for i, suggestion in enumerate(error_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"userguide-error-{file_name}-{i}",
                                    category="userguide.error_handling",
                                    severity="should_fix",
                                    source={"section": "userguide", "field": "error_handling_suggestions", "score": error_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint=f"Error-Handling-{i+1}",
                                    content_guidance=suggestion,
                                ))

                # If consistency issues present, add targeted improvements
                if isinstance(consistency_eval, dict):
                    score = consistency_eval.get("score")
                    if score in ("Poor", "Fair"):
                        suggestions.append(SuggestionItem(
                            id=f"userguide-consistency-{file_name}",
                            category="userguide.consistency",
                            severity="should_fix",
                            source={"section": "userguide", "field": "consistency", "score": score},
                            target_files=[file_name],
                            action="full_replace",
                            anchor_hint="Examples",
                            content_guidance="Improve consistency in examples, terminology, and formatting based on evaluation report.",
                        ))

        # Tutorials/vignettes suggestions (new interface) - ONLY Poor/Fair scores
        if getattr(report, "tutorial_evaluation", None) and isinstance(report.tutorial_evaluation, dict):
            for file_name, eval_block in report.tutorial_evaluation.items():
                tut_eval = eval_block.get("tutorial_evaluation") if isinstance(eval_block, dict) else None
                consistency_eval = eval_block.get("consistency_evaluation") if isinstance(eval_block, dict) else None
                if isinstance(tut_eval, dict):
                    # Only extract suggestions for Poor/Fair scores
                    
                    # Readability suggestions - only if score is Poor/Fair
                    readability_score = tut_eval.get("readability_score", "")
                    readability_suggestions = tut_eval.get("readability_suggestions", [])
                    if readability_score in ("Poor", "Fair") and readability_suggestions:
                        for i, suggestion in enumerate(readability_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"tutorial-readability-{file_name}-{i}",
                                    category="tutorial.readability",
                                    severity="should_fix",
                                    source={"section": "tutorial", "field": "readability_suggestions", "score": readability_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint="Introduction",
                                    content_guidance=suggestion,
                                ))
                    
                    # Setup and dependencies suggestions - only if score is Poor/Fair
                    setup_score = tut_eval.get("setup_and_dependencies_score", "")
                    setup_suggestions = tut_eval.get("setup_and_dependencies_suggestions", [])
                    if setup_score in ("Poor", "Fair") and setup_suggestions:
                        for i, suggestion in enumerate(setup_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"tutorial-setup-{file_name}-{i}",
                                    category="tutorial.setup",
                                    severity="should_fix",
                                    source={"section": "tutorial", "field": "setup_and_dependencies_suggestions", "score": setup_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint="Setup",
                                    content_guidance=suggestion,
                                ))
                    
                    # Reproducibility suggestions - only if score is Poor/Fair
                    reproducibility_score = tut_eval.get("reproducibility_score", "")
                    reproducibility_suggestions = tut_eval.get("reproducibility_suggestions", [])
                    if reproducibility_score in ("Poor", "Fair") and reproducibility_suggestions:
                        for i, suggestion in enumerate(reproducibility_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"tutorial-reproducibility-{file_name}-{i}",
                                    category="tutorial.reproducibility",
                                    severity="should_fix",
                                    source={"section": "tutorial", "field": "reproducibility_suggestions", "score": reproducibility_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint="Setup",
                                    content_guidance=suggestion,
                                ))
                    
                    # Structure and navigation suggestions - only if score is Poor/Fair
                    structure_score = tut_eval.get("structure_and_navigation_score", "")
                    structure_suggestions = tut_eval.get("structure_and_navigation_suggestions", [])
                    if structure_score in ("Poor", "Fair") and structure_suggestions:
                        for i, suggestion in enumerate(structure_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"tutorial-structure-{file_name}-{i}",
                                    category="tutorial.structure",
                                    severity="should_fix",
                                    source={"section": "tutorial", "field": "structure_and_navigation_suggestions", "score": structure_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint="Introduction",
                                    content_guidance=suggestion,
                                ))
                    
                    # Executable code quality suggestions - only if score is Poor/Fair
                    code_score = tut_eval.get("executable_code_quality_score", "")
                    code_suggestions = tut_eval.get("executable_code_quality_suggestions", [])
                    if code_score in ("Poor", "Fair") and code_suggestions:
                        for i, suggestion in enumerate(code_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"tutorial-code-{file_name}-{i}",
                                    category="tutorial.code_quality",
                                    severity="should_fix",
                                    source={"section": "tutorial", "field": "executable_code_quality_suggestions", "score": code_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint="Code Examples",
                                    content_guidance=suggestion,
                                ))
                    
                    # Result verification suggestions - only if score is Poor/Fair
                    verification_score = tut_eval.get("result_verification_score", "")
                    verification_suggestions = tut_eval.get("result_verification_suggestions", [])
                    if verification_score in ("Poor", "Fair") and verification_suggestions:
                        for i, suggestion in enumerate(verification_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"tutorial-verification-{file_name}-{i}",
                                    category="tutorial.verification",
                                    severity="should_fix",
                                    source={"section": "tutorial", "field": "result_verification_suggestions", "score": verification_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint="Results",
                                    content_guidance=suggestion,
                                ))
                    
                    # Performance and resource notes suggestions - only if score is Poor/Fair
                    performance_score = tut_eval.get("performance_and_resource_notes_score", "")
                    performance_suggestions = tut_eval.get("performance_and_resource_notes_suggestions", [])
                    if performance_score in ("Poor", "Fair") and performance_suggestions:
                        for i, suggestion in enumerate(performance_suggestions):
                            if isinstance(suggestion, str) and suggestion.strip():
                                suggestions.append(SuggestionItem(
                                    id=f"tutorial-performance-{file_name}-{i}",
                                    category="tutorial.performance",
                                    severity="should_fix",
                                    source={"section": "tutorial", "field": "performance_and_resource_notes_suggestions", "score": performance_score},
                                    target_files=[file_name],
                                    action="full_replace",
                                    anchor_hint="Performance",
                                    content_guidance=suggestion,
                                ))
                if isinstance(consistency_eval, dict):
                    score = consistency_eval.get("score")
                    if score in ("Poor", "Fair"):
                        suggestions.append(SuggestionItem(
                            id=f"tutorial-consistency-{file_name}",
                            category="tutorial.consistency",
                            severity="should_fix",
                            source={"section": "tutorial", "field": "consistency", "score": score},
                            target_files=[file_name],
                            action="full_replace",
                            anchor_hint=None,
                            content_guidance="Align tutorial with code definitions; fix inconsistencies as per report.",
                        ))

        return suggestions


