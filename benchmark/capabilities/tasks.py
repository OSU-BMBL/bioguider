"""Task definitions + ground truth for the capability benchmark.

Two task families:

* ``TOOL_TASKS``   — bind tools to the model, check selection / arguments / abstention.
* ``STRUCT_TASKS`` — ask for a typed object, check schema validity / field accuracy.

Everything here is data + light helpers; the scoring/invocation lives in
``runner.py``. Tasks are biomedical-flavored so the signal transfers to how
BioGuider uses these models in practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field


# ===========================================================================
# Tool schemas (bound via ChatOpenAI.bind_tools)
# ===========================================================================

class SearchGene(BaseModel):
    """Look up a gene by its official symbol in a reference database."""
    symbol: str = Field(description="Official gene symbol, e.g. TP53")
    species: str = Field(description="Species common name, e.g. human, mouse")


class InstallPackage(BaseModel):
    """Install a software package from a named registry."""
    name: str = Field(description="Package name")
    registry: str = Field(description="One of: pypi, cran, bioconductor, conda")


class ConvertUnits(BaseModel):
    """Convert a numeric quantity between two units."""
    value: float = Field(description="The numeric value to convert")
    from_unit: str = Field(description="Source unit, e.g. uL")
    to_unit: str = Field(description="Target unit, e.g. mL")


class RunAlignment(BaseModel):
    """Align sequencing reads against a reference genome."""
    reads_path: str = Field(description="Path to the FASTQ reads file")
    reference: str = Field(description="Reference genome identifier, e.g. GRCh38")
    threads: int = Field(description="Number of CPU threads to use")


# A standing tool palette bound on every tool task, so models must *select*,
# not just fill the single offered tool.
TOOL_PALETTE: List[Type[BaseModel]] = [SearchGene, InstallPackage, ConvertUnits, RunAlignment]


# ===========================================================================
# Structured-output schemas (via with_structured_output)
# ===========================================================================

class PackageMeta(BaseModel):
    """Flat extraction."""
    name: str
    language: str
    version: str


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class IssueReport(BaseModel):
    """Enum-constrained extraction."""
    title: str
    severity: Severity
    is_blocking: bool


class Author(BaseModel):
    name: str
    affiliation: str


class Citation(BaseModel):
    """Nested object."""
    paper_title: str
    year: int
    first_author: Author


class Dependency(BaseModel):
    name: str
    min_version: str


class DependencyList(BaseModel):
    """List-of-objects extraction."""
    dependencies: List[Dependency]


class InstallStep(BaseModel):
    """Optional fields — must omit/null what isn't stated."""
    command: str
    requires_sudo: bool
    note: Optional[str] = None


# ===========================================================================
# Task containers
# ===========================================================================

@dataclass
class ToolTask:
    id: str
    category: str            # selection | arguments | abstention | multi_step
    prompt: str
    expected_tool: Optional[str]          # None => model should NOT call a tool
    expected_args: Dict[str, Any] = field(default_factory=dict)
    tools: List[Type[BaseModel]] = field(default_factory=lambda: list(TOOL_PALETTE))


@dataclass
class StructTask:
    id: str
    category: str            # flat | enum | nested | list | optional
    prompt: str
    schema: Type[BaseModel] = None
    expected: Dict[str, Any] = field(default_factory=dict)


# --- argument / field comparison helpers -----------------------------------

def norm(v: Any) -> Any:
    """Loose normalization so 'Human' == 'human ' == 'human'."""
    if isinstance(v, str):
        return v.strip().lower()
    return v


def args_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """All expected keys present and loosely equal. Extra actual keys ignored."""
    for k, want in expected.items():
        if k not in actual:
            return False
        if norm(actual[k]) != norm(want):
            return False
    return True


# ===========================================================================
# Tool tasks
# ===========================================================================

TOOL_TASKS: List[ToolTask] = [
    ToolTask(
        id="tool_select_gene",
        category="selection",
        prompt="What is the function of the human gene TP53?",
        expected_tool="SearchGene",
        expected_args={"symbol": "TP53", "species": "human"},
    ),
    ToolTask(
        id="tool_select_install",
        category="selection",
        prompt="Please install the Seurat package from CRAN.",
        expected_tool="InstallPackage",
        expected_args={"name": "Seurat", "registry": "cran"},
    ),
    ToolTask(
        id="tool_args_convert",
        category="arguments",
        prompt="Convert 250 microliters to milliliters.",
        expected_tool="ConvertUnits",
        expected_args={"value": 250, "from_unit": "uL", "to_unit": "mL"},
    ),
    ToolTask(
        id="tool_args_alignment",
        category="arguments",
        prompt="Align the reads in /data/sample.fastq to GRCh38 using 8 threads.",
        expected_tool="RunAlignment",
        expected_args={"reads_path": "/data/sample.fastq", "reference": "GRCh38", "threads": 8},
    ),
    ToolTask(
        id="tool_disambiguate_bioc",
        category="selection",
        prompt="I need DESeq2 set up from Bioconductor.",
        expected_tool="InstallPackage",
        expected_args={"name": "DESeq2", "registry": "bioconductor"},
    ),
    # Abstention: no tool fits the request.
    ToolTask(
        id="tool_abstain_chitchat",
        category="abstention",
        prompt="Thanks, that's all for now — have a good day!",
        expected_tool=None,
    ),
    # Abstention: matching tool exists but a required arg is missing — a good
    # model should ask for the species rather than fabricate one.
    ToolTask(
        id="tool_abstain_underspecified",
        category="abstention",
        prompt="Look up the gene BRCA1.",  # species not given
        expected_tool=None,
    ),
    ToolTask(
        id="tool_args_mouse_gene",
        category="arguments",
        prompt="Find information about the mouse gene Trp53.",
        expected_tool="SearchGene",
        expected_args={"symbol": "Trp53", "species": "mouse"},
    ),
]


# ===========================================================================
# Structured-output tasks
# ===========================================================================

STRUCT_TASKS: List[StructTask] = [
    StructTask(
        id="struct_flat_meta",
        category="flat",
        schema=PackageMeta,
        prompt=(
            "Extract package metadata from this line:\n"
            "'scanpy is a Python toolkit, current release 1.10.2.'"
        ),
        expected={"name": "scanpy", "language": "Python", "version": "1.10.2"},
    ),
    StructTask(
        id="struct_enum_issue",
        category="enum",
        schema=IssueReport,
        prompt=(
            "Classify this issue:\n"
            "'Installation crashes on macOS — completely blocks all users from "
            "getting started.' Title it 'macOS install crash'."
        ),
        expected={"title": "macOS install crash", "severity": "high", "is_blocking": True},
    ),
    StructTask(
        id="struct_nested_citation",
        category="nested",
        schema=Citation,
        prompt=(
            "Extract the citation:\n"
            "'Stuart et al. (2019) Comprehensive Integration of Single-Cell Data. "
            "First author Tim Stuart, affiliation New York Genome Center.'"
        ),
        expected={
            "paper_title": "Comprehensive Integration of Single-Cell Data",
            "year": 2019,
            "first_author": {"name": "Tim Stuart", "affiliation": "New York Genome Center"},
        },
    ),
    StructTask(
        id="struct_list_deps",
        category="list",
        schema=DependencyList,
        prompt=(
            "List the dependencies and their minimum versions:\n"
            "'Requires numpy >= 1.21, pandas >= 1.3, and scipy >= 1.7.'"
        ),
        expected={
            "dependencies": [
                {"name": "numpy", "min_version": "1.21"},
                {"name": "pandas", "min_version": "1.3"},
                {"name": "scipy", "min_version": "1.7"},
            ]
        },
    ),
    StructTask(
        id="struct_optional_present",
        category="optional",
        schema=InstallStep,
        prompt=(
            "Extract the install step:\n"
            "'Run `sudo apt-get install samtools`. Note: requires Ubuntu 20.04+.'"
        ),
        expected={
            "command": "sudo apt-get install samtools",
            "requires_sudo": True,
            "note": "requires Ubuntu 20.04+",
        },
    ),
    StructTask(
        id="struct_optional_absent",
        category="optional",
        schema=InstallStep,
        prompt=(
            "Extract the install step:\n"
            "'Run `pip install pharokka`.'"  # no sudo, no note
        ),
        expected={"command": "pip install pharokka", "requires_sudo": False, "note": None},
    ),
]
