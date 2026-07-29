"""
Rd file tests.
"""
import os
import re
from pathlib import Path
from datetime import datetime

import pytest

from benchmark.shared import *


# ============================================================================
# .Rd FILE GENERATOR VERIFICATION
# ============================================================================

def test_rd_file_generator(llm, test_output_dir):
    r"""
    Verify BioGuider generator behaviour on .Rd (R documentation) files.

    Manually injects prose-only errors into RunPCA.Rd (description, arguments,
    value sections) then runs evaluate_and_refine_document() and checks:
      1. Prose errors in \description{} / \arguments{} / \value{} are fixed.
      2. \usage{} block is NOT modified (roxygen2-generated, must be pristine).
      3. Output retains valid .Rd structure (\name, \alias, \usage, \arguments).

    Run:
        pytest system_tests/test_single_file_stress.py::test_rd_file_generator -v -s
    """
    import shutil
    from bioguider.generation.document_pipeline import DocumentPipeline
    from bioguider.utils.constants import EvaluationTypeEnum

    RD_FILE = "data/.adalflow/repos/satijalab_seurat/man/RunPCA.Rd"
    if not os.path.exists(RD_FILE):
        import pytest
        pytest.skip(f".Rd file not found: {RD_FILE}")

    original = Path(RD_FILE).read_text(encoding="utf-8")

    # ── Manually inject prose errors (prose sections only, no \usage{}) ────
    corrupted = original
    injected_errors = []

    # \description{} — add a typo and a scientific error
    OLD_DESC = "Run a PCA dimensionality reduction."
    NEW_DESC = "Run a PCA diminsionality reducton."
    if OLD_DESC in corrupted:
        corrupted = corrupted.replace(OLD_DESC, NEW_DESC, 1)
        injected_errors.append(("description typo", OLD_DESC, NEW_DESC))

    # \arguments{} \item{npcs} — wrong number
    OLD_NPCS = "Total Number of PCs to compute and store (50 by default)"
    NEW_NPCS = "Total Number of PCs to compute and store (100 by default)"
    if OLD_NPCS in corrupted:
        corrupted = corrupted.replace(OLD_NPCS, NEW_NPCS, 1)
        injected_errors.append(("argument number error", OLD_NPCS, NEW_NPCS))

    # \arguments{} \item{seed.use} — wrong value
    OLD_SEED = "sets the seed to 42"
    NEW_SEED = "sets the seed to 99"
    if OLD_SEED in corrupted:
        corrupted = corrupted.replace(OLD_SEED, NEW_SEED, 1)
        injected_errors.append(("seed value error", OLD_SEED, NEW_SEED))

    # \value{} — broken prose
    OLD_VAL = "Returns Seurat object with the PCA calculation stored in the reductions slot"
    NEW_VAL = "Returns Seurat obejct with PCA calcualtion store in reductions slot"
    if OLD_VAL in corrupted:
        corrupted = corrupted.replace(OLD_VAL, NEW_VAL, 1)
        injected_errors.append(("value typos", OLD_VAL, NEW_VAL))

    assert injected_errors, "No errors were injected — check .Rd content"
    print(f"\nInjected {len(injected_errors)} prose errors into RunPCA.Rd:")
    for name, orig, mut in injected_errors:
        print(f"  [{name}] {repr(orig)} → {repr(mut)}")

    run_root = os.path.join("outputs/rd_generator_test", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_root, exist_ok=True)

    # Write original and corrupted files
    write_file(os.path.join(run_root, "RunPCA.original.Rd"), original)
    corrupted_filename = "RunPCA.corrupted.Rd"
    write_file(os.path.join(run_root, corrupted_filename), corrupted)

    # ── Run BioGuider pipeline ─────────────────────────────────────────────
    print(f"\nBuilding DocumentPipeline from {SEURAT_REPO_PATH} ...")
    pipeline = DocumentPipeline(SEURAT_REPO_PATH).prepare_repo(llm)

    report_path = os.path.join(run_root, "RunPCA.eval_report.json")
    merged_report, fixed_content = pipeline.evaluate_and_refine_document(
        llm=llm,
        doc_repo_path=run_root,
        doc_path=corrupted_filename,
        eval_type=EvaluationTypeEnum.USERGUIDE,
        report_output_path=report_path,
    )
    write_file(os.path.join(run_root, "RunPCA.fixed.Rd"), fixed_content)

    suggestions = merged_report.get("total_suggestions", 0)
    print(f"\nEval report: {suggestions} suggestions")
    for s in merged_report.get("suggestions", []):
        print(f"  [{s['category']}] {s['content_guidance'][:100]}")

    # ── Checks ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")

    # 1. Did it fix prose errors?
    prose_fixed = 0
    for name, orig, mut in injected_errors:
        was_in_corrupted = mut in corrupted
        is_in_fixed = mut in fixed_content
        is_orig_restored = orig in fixed_content
        status = "FIXED" if (is_orig_restored or not is_in_fixed) else "UNFIXED"
        print(f"  [{status}] {name}: mutated={repr(mut[:40])}")
        if status == "FIXED":
            prose_fixed += 1
    print(f"  Prose errors fixed: {prose_fixed}/{len(injected_errors)}")

    # 2. Was \usage{} preserved?
    orig_usage = re.search(r"\\usage\{(.+?)^\}", original, re.DOTALL | re.MULTILINE)
    fixed_usage = re.search(r"\\usage\{(.+?)^\}", fixed_content, re.DOTALL | re.MULTILINE)
    if orig_usage and fixed_usage:
        usage_preserved = orig_usage.group(0) == fixed_usage.group(0)
        print(f"\n  \\usage{{}} preserved: {usage_preserved}")
    else:
        usage_preserved = False
        print(f"\n  \\usage{{}} block: orig_found={bool(orig_usage)}, fixed_found={bool(fixed_usage)}")

    # 3. Valid .Rd structure?
    rd_keywords = [r"\name{", r"\alias{", r"\usage{", r"\arguments{", r"\description{"]
    structure_ok = all(kw in fixed_content for kw in rd_keywords)
    print(f"  .Rd structure intact (all key tags present): {structure_ok}")
    for kw in rd_keywords:
        print(f"    {kw}: {'✓' if kw in fixed_content else '✗ MISSING'}")

    print(f"\nArtifacts: {run_root}")


# ============================================================================
# DocumentationGenerationManager behaviour on .Rd files
# ============================================================================

def test_generation_manager_on_rd_file(llm, test_output_dir):
    r"""
    Verify DocumentationGenerationManager behaviour when an evaluation report
    references a .Rd file (R package documentation).

    Injects errors into three distinct zones of RunPCA.Rd:
      Zone A — prose sections (\description, \arguments, \value)
              These SHOULD be fixed by the generator.
      Zone B — \usage{} block (roxygen2-generated function signatures)
              Must NOT be modified; any change breaks the package.
      Zone C — \examples{} block (executable R code run by R CMD check)
              Should NOT be modified; changes risk breaking R CMD check.

    Steps:
      1. Copy RunPCA.Rd into a temp repo dir and inject errors into all three zones.
      2. Write a minimal evaluation report JSON that lists the .Rd file under
         userguide_files so DocumentationGenerationManager picks it up.
      3. Run DocumentationGenerationManager.run().
      4. Assert per-zone outcomes and print a clear pass/fail table.

    Run:
        pytest system_tests/test_single_file_stress.py::test_generation_manager_on_rd_file -v -s
    """
    import shutil
    import tempfile
    from bioguider.managers.generation_manager import DocumentationGenerationManager
    from bioguider.managers.config import GenerationConfig

    RD_SRC = "data/.adalflow/repos/satijalab_seurat/man/RunPCA.Rd"
    if not os.path.exists(RD_SRC):
        import pytest
        pytest.skip(f".Rd file not found: {RD_SRC}")

    original = Path(RD_SRC).read_text(encoding="utf-8")

    # Verify the file has \examples{} — pick one that does, or skip that zone
    has_examples = r"\examples{" in original

    run_root = os.path.join(
        "outputs/rd_generation_manager_test",
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_root, exist_ok=True)

    # ── Build temp repo dir with a single .Rd file ─────────────────────────
    tmp_repo = os.path.join(run_root, "tmp_repo")
    man_dir = os.path.join(tmp_repo, "man")
    os.makedirs(man_dir, exist_ok=True)

    corrupted = original
    injected = {}   # zone -> (original_text, mutated_text)

    # Zone A — prose: \description{}
    A1_ORIG = "Run a PCA dimensionality reduction."
    A1_MUT  = "Run a PCA diminsionality reducton."
    if A1_ORIG in corrupted:
        corrupted = corrupted.replace(A1_ORIG, A1_MUT, 1)
        injected["prose_description"] = (A1_ORIG, A1_MUT)

    # Zone A — prose: \arguments \item{npcs}
    A2_ORIG = "Total Number of PCs to compute and store (50 by default)"
    A2_MUT  = "Total Number of PCs to compuet and stroe (50 by default)"
    if A2_ORIG in corrupted:
        corrupted = corrupted.replace(A2_ORIG, A2_MUT, 1)
        injected["prose_arguments"] = (A2_ORIG, A2_MUT)

    # Zone A — prose: \value{}
    A3_ORIG = "Returns Seurat object with the PCA calculation stored in the reductions slot"
    A3_MUT  = "Returns Seurat obejct with the PCA calculaton stored in the reductions solt"
    if A3_ORIG in corrupted:
        corrupted = corrupted.replace(A3_ORIG, A3_MUT, 1)
        injected["prose_value"] = (A3_ORIG, A3_MUT)

    # Zone B — \usage{}: corrupt one argument default value inside the block
    # We find "npcs = 50" inside \usage and change it to "npcs = 99"
    import re as _re
    usage_match = _re.search(r"(\\usage\{.*?^\})", corrupted, _re.DOTALL | _re.MULTILINE)
    B_ORIG_USAGE = None
    B_MUT_USAGE  = None
    if usage_match:
        block = usage_match.group(1)
        if "npcs = 50" in block:
            B_ORIG_USAGE = block
            B_MUT_USAGE  = block.replace("npcs = 50", "npcs = 99")
            corrupted = corrupted.replace(B_ORIG_USAGE, B_MUT_USAGE, 1)
            injected["usage_block"] = ("npcs = 50  (inside \\usage{})", "npcs = 99")

    # Zone C — \examples{}: if present, add a typo in a function call
    C_ORIG = None
    C_MUT  = None
    if has_examples:
        ex_match = _re.search(r"(\\examples\{.*?^\})", corrupted, _re.DOTALL | _re.MULTILINE)
        if ex_match:
            block = ex_match.group(1)
            # find first function call pattern like FunctionName(
            fn_match = _re.search(r"([A-Z][A-Za-z]+)\(", block)
            if fn_match:
                fname = fn_match.group(1)
                mid = len(fname) // 2
                if mid + 2 <= len(fname) and fname[mid] != fname[mid + 1]:
                    mutated = fname[:mid] + fname[mid+1] + fname[mid] + fname[mid+2:]
                    if mutated != fname:
                        C_ORIG = fname + "("
                        C_MUT  = mutated + "("
                        new_corrupted = corrupted.replace(
                            ex_match.group(1),
                            ex_match.group(1).replace(C_ORIG, C_MUT, 1),
                            1,
                        )
                        if new_corrupted != corrupted:
                            corrupted = new_corrupted
                            injected["examples_block"] = (C_ORIG, C_MUT)

    print(f"\nInjected errors into {len(injected)} zones:")
    for zone, (orig, mut) in injected.items():
        print(f"  [{zone}] {repr(orig[:60])} → {repr(mut[:60])}")

    rd_filename = "RunPCA.Rd"
    rd_path = os.path.join(man_dir, rd_filename)
    Path(rd_path).write_text(corrupted, encoding="utf-8")
    write_file(os.path.join(run_root, "RunPCA.corrupted.Rd"), corrupted)
    write_file(os.path.join(run_root, "RunPCA.original.Rd"), original)

    # ── Write evaluation report JSON ───────────────────────────────────────
    # Format expected by EvaluationReportLoader.load():
    #   userguide.files  → list of relative paths from repo root
    #   userguide.evaluation → dict (content drives SuggestionExtractor)
    rel_rd_path = os.path.join("man", rd_filename)
    # SuggestionExtractor expects userguide_evaluation keyed by filename, each
    # value containing "user_guide_evaluation" with score+suggestion list fields.
    report_data = {
        "repo_url": tmp_repo,
        "userguide": {
            "files": [rel_rd_path],
            "evaluation": {
                rel_rd_path: {
                    "user_guide_evaluation": {
                        "readability_score": "Poor",
                        "readability_suggestions": [
                            "Fix typos in description: 'diminsionality' should be 'dimensionality', 'reducton' should be 'reduction'.",
                            "Fix typos in arguments item npcs: 'compuet' should be 'compute', 'stroe' should be 'store'.",
                            "Fix typos in value: 'obejct' should be 'object', 'calculaton' should be 'calculation', 'solt' should be 'slot'.",
                        ],
                        "context_and_purpose_score": "Poor",
                        "context_and_purpose_suggestions": [
                            "Expand the description to explain why PCA is used in single-cell analysis.",
                        ],
                        "error_handling_score": "Poor",
                        "error_handling_suggestions": [
                            "Document what happens when features are not found in the scaled data.",
                        ],
                    }
                }
            },
        },
        "installation": {"evaluation": {}, "files": []},
        "readme": {"evaluations": {}, "files": []},
    }
    report_path = os.path.join(run_root, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # ── Run DocumentationGenerationManager ────────────────────────────────
    print(f"\nRunning DocumentationGenerationManager on {rd_filename} ...")
    config = GenerationConfig(
        debug_output=False,
        clean_output=False,
        write_originals=True,
        max_files=1,
        target_files=[rel_rd_path],
    )
    gen = DocumentationGenerationManager(llm, step_callback=None, config=config)
    out_dir = gen.run(
        report_path=report_path,
        repo_path=tmp_repo,
        target_files=[rel_rd_path],
        max_files=1,
    )
    print(f"Generation output dir: {out_dir}")

    # ── Load fixed file ────────────────────────────────────────────────────
    fixed_rd = os.path.join(out_dir, "man", rd_filename)
    if not os.path.exists(fixed_rd):
        # try flat output dir
        fixed_rd = os.path.join(out_dir, rd_filename)
    if not os.path.exists(fixed_rd):
        # search recursively
        candidates = list(Path(out_dir).rglob(rd_filename))
        fixed_rd = str(candidates[0]) if candidates else None

    assert fixed_rd and os.path.exists(fixed_rd), (
        f"Generator did not produce {rd_filename}. out_dir contents: "
        + str(list(Path(out_dir).rglob("*")))
    )
    fixed = Path(fixed_rd).read_text(encoding="utf-8")
    write_file(os.path.join(run_root, "RunPCA.fixed.Rd"), fixed)

    # ── Evaluate per-zone outcomes ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("ZONE-BY-ZONE RESULTS")
    print(f"{'='*65}")

    results = {}

    # Zone A — prose errors: expect fixed
    for zone, (orig, mut) in injected.items():
        if not zone.startswith("prose_"):
            continue
        mut_still_present = mut in fixed
        orig_restored = orig in fixed
        status = "FIXED" if (orig_restored or not mut_still_present) else "UNFIXED"
        results[zone] = status
        print(f"  Zone A [{zone}]: {status}")
        print(f"    mutated text present in output: {mut_still_present}")
        print(f"    original text restored:         {orig_restored}")

    # Zone B — \usage{} block: expect UNCHANGED
    if "usage_block" in injected:
        orig_usage_block, _ = injected["usage_block"]
        # Extract \usage{} from fixed output
        fixed_usage_match = _re.search(r"(\\usage\{.*?^\})", fixed, _re.DOTALL | _re.MULTILINE)
        if fixed_usage_match:
            fixed_block = fixed_usage_match.group(1)
            usage_mutated = "npcs = 99" in fixed_block
            usage_original = "npcs = 50" in fixed_block
            if usage_original:
                status = "PRESERVED (correctly reverted or untouched)"
            elif usage_mutated:
                status = "MUTATED — injected error still present (not fixed)"
            else:
                status = "ALTERED — neither original nor mutated value found"
            results["usage_block"] = status
            print(f"\n  Zone B [\\usage{{}}]: {status}")
        else:
            results["usage_block"] = "MISSING — \\usage{} block not found in output"
            print(f"\n  Zone B [\\usage{{}}]: MISSING from output")

    # Zone C — \examples{} block: expect UNCHANGED
    if "examples_block" in injected:
        orig_fn, mut_fn = injected["examples_block"]
        ex_match_fixed = _re.search(r"(\\examples\{.*?^\})", fixed, _re.DOTALL | _re.MULTILINE)
        if ex_match_fixed:
            fixed_ex_block = ex_match_fixed.group(1)
            ex_mut_present = mut_fn in fixed_ex_block
            ex_orig_present = orig_fn in fixed_ex_block
            if ex_orig_present and not ex_mut_present:
                status = "PRESERVED (correctly reverted or untouched)"
            elif ex_mut_present:
                status = "MUTATED — injected error still present (not fixed)"
            else:
                status = "ALTERED — neither original nor mutated text found"
            results["examples_block"] = status
            print(f"  Zone C [\\examples{{}}]: {status}")
        else:
            results["examples_block"] = "NO \\examples{} block in output"
            print(f"  Zone C [\\examples{{}}]: no \\examples{{}} block in output")

    # .Rd structure check
    rd_tags = [r"\name{", r"\usage{", r"\arguments{", r"\description{"]
    structure_ok = all(tag in fixed for tag in rd_tags)
    print(f"\n  .Rd structure intact: {structure_ok}")

    print(f"\nArtifacts: {run_root}")
    print(f"  corrupted:  RunPCA.corrupted.Rd")
    print(f"  eval report: eval_report.json")
    print(f"  fixed:      RunPCA.fixed.Rd")
