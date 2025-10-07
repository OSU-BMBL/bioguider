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

    def run_quant_test(self, report_path: str, baseline_repo_path: str, tmp_repo_path: str, min_per_category: int = 3) -> str:
        self.print_step("QuantTest:LoadBaseline", baseline_repo_path)
        baseline_readme_path = os.path.join(baseline_repo_path, "README.md")
        baseline = read_file(baseline_readme_path) or ""

        self.print_step("QuantTest:Inject")
        injector = LLMErrorInjector(self.llm)
        corrupted, inj_manifest = injector.inject(baseline, min_per_category=min_per_category)

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
        # slides-like markdown report
        totals = results.get("summary", {}).get("totals", {})
        success_rate = results.get("summary", {}).get("success_rate", 0.0)
        lines = ["# 🔬 Quantifiable Testing Results\n",
                 "\n## BioGuider Error Correction Performance Analysis\n",
                 "\n---\n",
                 "\n## 📊 Slide 1: Testing Results Overview\n",
                 "\n### 🎯 Totals\n",
                 f"- Total Errors: {totals.get('total_errors', 0)}\n",
                 f"- Fixed to Baseline: {totals.get('fixed_to_baseline', 0)}\n",
                 f"- Fixed to Valid: {totals.get('fixed_to_valid', 0)}\n",
                 f"- Unchanged: {totals.get('unchanged', 0)}\n",
                 f"- Success Rate: {success_rate}%\n",
                 "\n### 📂 Per-Category Metrics\n"]
        for cat, m in results["per_category"].items():
            lines.append(f"- {cat}: total={m.get('total',0)}, fixed_to_baseline={m.get('fixed_to_baseline',0)}, fixed_to_valid={m.get('fixed_to_valid',0)}, unchanged={m.get('unchanged',0)}")
        # Per-file change counts (simple heuristic from manifest artifacts)
        try:
            manifest_path = os.path.join(out_dir, "manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as mf:
                mani = json.load(mf)
            lines.append("\n### 🗂️ Per-File Changes\n")
            for art in mani.get("artifacts", []):
                rel = art.get("dest_rel_path")
                stats = art.get("diff_stats", {})
                added = stats.get("added_lines", 0)
                status = "Revised" if added and added > 0 else "Copied"
                lines.append(f"- {rel}: {status}, added_lines={added}")
        except Exception:
            pass
        lines.append("\n---\n\n## 📝 Notes\n")
        lines.append("- README versions saved: README.original.md, README.corrupted.md, README.md (fixed).\n")
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

    def run_quant_suite(self, report_path: str, baseline_repo_path: str, base_tmp_repo_path: str, levels: dict[str, int]) -> dict:
        results = {}
        for level, min_cnt in levels.items():
            tmp_repo_path = f"{base_tmp_repo_path}_{level}"
            out_dir = self.run_quant_test(report_path, baseline_repo_path, tmp_repo_path, min_per_category=min_cnt)
            results[level] = out_dir
        return results


