from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    timestamp: Optional[str] = None
    repo_url: Optional[str] = None

    installation_evaluation: Optional[Dict[str, Any]] = None
    installation_files: Optional[List[str]] = None

    readme_evaluation: Optional[Dict[str, Any]] = None
    readme_files: Optional[List[str]] = None

    # Optional: rich user guide evaluation content and any explicitly listed files
    userguide_evaluation: Optional[Dict[str, Any]] = None
    userguide_files: Optional[List[str]] = None

    # Optional: tutorial evaluation content and any explicitly listed files
    tutorial_evaluation: Optional[Dict[str, Any]] = None
    tutorial_files: Optional[List[str]] = None

    submission_requirements_evaluation: Optional[Dict[str, Any]] = None
    submission_requirements_files: Optional[List[str]] = None


class SuggestionItem(BaseModel):
    id: str
    category: str
    severity: str = Field(default="should_fix")
    source: Dict[str, str] = Field(default_factory=dict)
    target_files: List[str] = Field(default_factory=list)
    action: str
    anchor_hint: Optional[str] = None
    content_guidance: Optional[str] = None


class StyleProfile(BaseModel):
    heading_style: str = Field(default="#")
    list_style: str = Field(default="-")
    code_fence_style: str = Field(default="```")
    tone_markers: List[str] = Field(default_factory=list)
    link_style: str = Field(default="inline")


class PlannedEdit(BaseModel):
    file_path: str
    edit_type: str
    anchor: Dict[str, str] = Field(default_factory=dict)
    content_template: str
    rationale: str
    minimal_diff: bool = Field(default=True)
    suggestion_id: Optional[str] = None


class DocumentPlan(BaseModel):
    repo_path: str
    style_profile: StyleProfile
    planned_edits: List[PlannedEdit] = Field(default_factory=list)


class OutputArtifact(BaseModel):
    dest_rel_path: str
    original_rel_path: str
    change_summary: str
    diff_stats: Dict[str, int] = Field(default_factory=dict)


class GenerationManifest(BaseModel):
    repo_url: Optional[str] = None
    report_path: Optional[str] = None
    output_dir: Optional[str] = None
    suggestions: List[SuggestionItem] = Field(default_factory=list)
    planned_edits: List[PlannedEdit] = Field(default_factory=list)
    artifacts: List[OutputArtifact] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)

class GenerationReport(BaseModel):
    repo_url: Optional[str] = None
    output_dir: Optional[str] = None
    sections: List[Dict[str, Any]] = Field(default_factory=list)


