from .models import (
    EvaluationReport,
    SuggestionItem,
    StyleProfile,
    PlannedEdit,
    DocumentPlan,
    OutputArtifact,
    GenerationManifest,
)
from .report_loader import EvaluationReportLoader
from .suggestion_extractor import SuggestionExtractor
from .repo_reader import RepoReader
from .style_analyzer import StyleAnalyzer
from .change_planner import ChangePlanner
from .document_renderer import DocumentRenderer
from .output_manager import OutputManager
from .llm_content_generator import LLMContentGenerator
from .llm_cleaner import LLMCleaner

# New unified modules (Phase 1 refactoring)
from .constants import (
    EvaluationScore,
    FileCategory,
    ErrorCategory,
    FixStatus,
    EditType,
    GenerationLimits,
    CANONICAL_README_TITLES,
)
from .config import (
    LLMGenerationConfig,
    InjectionConfig,
    MetricsConfig,
)
from .unified_metrics import (
    UnifiedMetricsEvaluator,
    EvaluationResult,
    ErrorEvaluation,
    CategoryMetrics,
    evaluate_fixes,  # Backward compatible wrapper
)

__all__ = [
    # Models
    "EvaluationReport",
    "SuggestionItem",
    "StyleProfile",
    "PlannedEdit",
    "DocumentPlan",
    "OutputArtifact",
    "GenerationManifest",
    # Pipeline components
    "EvaluationReportLoader",
    "SuggestionExtractor",
    "RepoReader",
    "StyleAnalyzer",
    "ChangePlanner",
    "DocumentRenderer",
    "OutputManager",
    "LLMContentGenerator",
    "LLMCleaner",
    # Constants
    "EvaluationScore",
    "FileCategory",
    "ErrorCategory",
    "FixStatus",
    "EditType",
    "GenerationLimits",
    "CANONICAL_README_TITLES",
    # Config
    "LLMGenerationConfig",
    "InjectionConfig",
    "MetricsConfig",
    # Metrics
    "UnifiedMetricsEvaluator",
    "EvaluationResult",
    "ErrorEvaluation",
    "CategoryMetrics",
    "evaluate_fixes",
]
