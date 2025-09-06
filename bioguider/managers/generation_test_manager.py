from __future__ import annotations

import os
import json
from typing import Tuple

from bioguider.generation.llm_injector import LLMErrorInjector
from bioguider.generation.test_metrics import evaluate_fixes
from bioguider.managers.generation_manager import DocumentationGenerationManager
from bioguider.agents.agent_utils import read_file, write_file


class GenerationTestManager:
    def __init__(self, llm, step_callback):
        self.llm = llm
        self.step_output = step_callback

    def print_step(self, name: str, out: str | None = None):
        if self.step_output:
            self.step_output(step_name=name, step_output=out)

    def run_quant_test(self, report_path: str, baseline_repo_path: str, tmp_repo_path: str) -> str:
        self.print_step("QuantTest:LoadBaseline", baseline_repo_path)
        baseline_readme_path = os.path.join(baseline_repo_path, "README.md")
        baseline = read_file(baseline_readme_path) or ""

        self.print_step("QuantTest:Inject")
        injector = LLMErrorInjector(self.llm)
        corrupted, inj_manifest = injector.inject(baseline, min_per_category=3)

        # write corrupted into tmp repo path
        os.makedirs(tmp_repo_path, exist_ok=True)
        corrupted_readme_path = os.path.join(tmp_repo_path, "README.md")
        write_file(corrupted_readme_path, corrupted)
        inj_path = os.path.join(tmp_repo_path, "INJECTION_MANIFEST.json")
        with open(inj_path, "w", encoding="utf-8") as fobj:
            json.dump(inj_manifest, fobj, indent=2)

        self.print_step("QuantTest:Generate")
        gen = DocumentationGenerationManager(self.llm, self.step_output)
        out_dir = gen.run(report_path=report_path, repo_path=tmp_repo_path)

        # read revised
        revised_readme_path = os.path.join(out_dir, "README.md")
        revised = read_file(revised_readme_path) or ""

        self.print_step("QuantTest:Evaluate")
        results = evaluate_fixes(baseline, corrupted, revised, inj_manifest)
        # write results
        with open(os.path.join(out_dir, "GEN_TEST_RESULTS.json"), "w", encoding="utf-8") as fobj:
            json.dump(results, fobj, indent=2)
        # simple md report
        lines = ["# Quantifiable Generation Test Report\n"]
        lines.append("## Metrics by Category\n")
        for cat, m in results["per_category"].items():
            lines.append(f"- {cat}: {m}")
        lines.append("\n## Notes\n")
        lines.append("- Three versions saved in this directory: README.original.md, README.corrupted.md, README.md (fixed).")
        with open(os.path.join(out_dir, "GEN_TEST_REPORT.md"), "w", encoding="utf-8") as fobj:
            fobj.write("\n".join(lines))
        # Save versioned files into output dir
        write_file(os.path.join(out_dir, "README.original.md"), baseline)
        write_file(os.path.join(out_dir, "README.corrupted.md"), corrupted)
        # Copy injection manifest
        try:
            with open(inj_path, "r", encoding="utf-8") as fin:
                with open(os.path.join(out_dir, "INJECTION_MANIFEST.json"), "w", encoding="utf-8") as fout:
                    fout.write(fin.read())
        except Exception:
            pass
        self.print_step("QuantTest:Done", out_dir)
        return out_dir


