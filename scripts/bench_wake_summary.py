"""Produce a compact wake-up briefing from an in-progress or completed multi_file_stress run.

Usage:
    poetry run python scripts/bench_wake_summary.py outputs/multi_file_stress/run_<ts>

Emits WAKE_SUMMARY.md at the run root with: files completed, mean F1_scorable by
model, top-line moat hits, pointers to per-file figures, failure ledger.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean


def load_per_file(run_dir: Path):
    per_file = {}
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir() or sub.name == "_aggregate":
            continue
        rpath = sub / "STRESS_TEST_RESULTS.json"
        if not rpath.exists():
            continue
        try:
            per_file[sub.name] = json.loads(rpath.read_text())
        except Exception as e:
            per_file[sub.name] = {"_load_error": str(e)}
    return per_file


def summarise(per_file: dict):
    model_f1s = defaultdict(list)
    model_f1s_scorable = defaultdict(list)
    total_rows = 0
    moat_cats = defaultdict(int)

    for file_stem, payload in per_file.items():
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            continue
        for r in results:
            total_rows += 1
            m = r.get("model", "?")
            f1 = r.get("f1_score")
            f1s = r.get("f1_score_scorable")
            if f1 is not None:
                model_f1s[m].append(f1)
            if f1s is not None:
                model_f1s_scorable[m].append(f1s)
            for cat in r.get("category_breakdown", []):
                cname = cat.get("category", "")
                if cname.startswith("prose_code_") or cname == "accession_id_prefix":
                    moat_cats[cname] += cat.get("injected", 0)

    summary = {
        "files_completed": len(per_file),
        "total_rows": total_rows,
        "mean_f1_by_model": {m: round(mean(vs), 3) for m, vs in sorted(model_f1s.items())},
        "mean_f1_scorable_by_model": {m: round(mean(vs), 3) for m, vs in sorted(model_f1s_scorable.items())},
        "moat_category_injection_counts": dict(moat_cats),
    }
    return summary


def figure_inventory(run_dir: Path):
    figs = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        for png in sorted(sub.glob("fig*.png")):
            figs.append(str(png.relative_to(run_dir)))
    return figs


def ensure_figures(run_dir: Path):
    """Idempotently render fig1-6 on any subdir with STRESS_TEST_RESULTS.json."""
    from bioguider.generation.viz import BenchmarkPlotter

    rendered, skipped = [], []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "STRESS_TEST_RESULTS.json").exists():
            continue
        if list(sub.glob("fig1_*.png")):
            skipped.append(sub.name)
            continue
        try:
            BenchmarkPlotter(sub).render_all()
            rendered.append(sub.name)
        except Exception as e:  # noqa: BLE001 — we log, don't fail
            rendered.append(f"{sub.name} (FAILED: {e!s})")
    return rendered, skipped


def main():
    if len(sys.argv) != 2:
        print("usage: bench_wake_summary.py <run_dir>")
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"run_dir not found: {run_dir}")
        sys.exit(2)

    per_file = load_per_file(run_dir)
    summary = summarise(per_file)
    rendered, skipped = ensure_figures(run_dir)
    if rendered:
        print(f"rendered figures in: {rendered}")
    if skipped:
        print(f"figures already present in: {skipped}")
    figs = figure_inventory(run_dir)

    md = []
    md.append("# Benchmark wake-up briefing\n")
    md.append(f"**Run dir**: `{run_dir}`\n")
    md.append(f"**Files completed**: {summary['files_completed']} / 10\n")
    md.append(f"**Total result rows**: {summary['total_rows']} (expected 450 at full completion)\n\n")

    md.append("## Mean F1 by model (headline / scorable)\n\n")
    md.append("| Model | Mean F1 | Mean F1 (scorable) |\n|---|---|---|\n")
    models = sorted(set(summary["mean_f1_by_model"]) | set(summary["mean_f1_scorable_by_model"]))
    for m in models:
        h = summary["mean_f1_by_model"].get(m, "-")
        s = summary["mean_f1_scorable_by_model"].get(m, "-")
        md.append(f"| `{m}` | {h} | {s} |\n")
    md.append("\n")

    if summary["moat_category_injection_counts"]:
        md.append("## prose_code_consistency moat — injected counts\n\n")
        for cat, n in sorted(summary["moat_category_injection_counts"].items()):
            md.append(f"- `{cat}`: {n} injected across runs\n")
    else:
        md.append("## prose_code_consistency moat\n\nNo moat-category injections landed. "
                  "(Likely means anchor regexes don't match Seurat idioms; covered in the earlier plan.)\n")
    md.append("\n")

    md.append(f"## Figures ({len(figs)} PNG)\n\n")
    if figs:
        for f in figs:
            md.append(f"- `{f}`\n")
    else:
        md.append("_No figures produced yet._\n")
    md.append("\n")

    md.append("## Per-file rows\n\n")
    md.append("| File | Rows | Model F1_scorable means |\n|---|---|---|\n")
    for stem, payload in per_file.items():
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            md.append(f"| `{stem}` | 0 | _no results_ |\n")
            continue
        per_model = defaultdict(list)
        for r in results:
            per_model[r.get("model", "?")].append(r.get("f1_score_scorable", 0.0))
        summary_line = ", ".join(f"{m}={round(mean(vs),2)}" for m, vs in sorted(per_model.items()))
        md.append(f"| `{stem}` | {len(results)} | {summary_line} |\n")
    md.append("\n")

    out_path = run_dir / "WAKE_SUMMARY.md"
    out_path.write_text("".join(md))
    print(f"wrote {out_path}")
    print(f"files_completed={summary['files_completed']}, total_rows={summary['total_rows']}")


if __name__ == "__main__":
    main()
