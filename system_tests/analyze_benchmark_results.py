import json
import os
from pathlib import Path
import argparse
from datetime import datetime

def analyze_benchmark(results_path):
    """
    Analyze the GEN_TEST_RESULTS.json and INJECTION_MANIFEST.json to produce a summary report.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"Error: {results_path} does not exist")
        return

    manifest_path = results_path.parent / "INJECTION_MANIFEST.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} does not exist")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Analyzing benchmark results from {results_path.parent}")
    
    # Aggregate stats
    total_files = len(results)
    total_injected = manifest.get("total_errors", 0)
    total_detected = 0
    
    file_stats = []
    
    for file_path, eval_result in results.items():
        # Get injected errors for this file
        injected_count = 0
        injected_categories = {}
        
        # Find file in manifest
        # Manifest keys might be relative paths
        manifest_file_info = manifest.get("files", {}).get(file_path, {})
        injected_count = manifest_file_info.get("error_count", 0)
        for err in manifest_file_info.get("errors", []):
            cat = err.get("category", "unknown")
            injected_categories[cat] = injected_categories.get(cat, 0) + 1
            
        # Get detected errors
        detected_count = 0
        detected_categories = {} # Hard to map exactly without detailed matching, but we can count
        
        # Check structured evaluation
        if "structured_evaluation" in eval_result and eval_result["structured_evaluation"]:
            struct = eval_result["structured_evaluation"]
            detected_count = struct.get("readability_error_count", 0)
            # Fallback if count is 0 but suggestions exist
            if detected_count == 0 and struct.get("readability_suggestions"):
                detected_count = len(struct.get("readability_suggestions"))
                
        # Check tutorial evaluation
        elif "tutorial_evaluation" in eval_result and eval_result["tutorial_evaluation"]:
            tut = eval_result["tutorial_evaluation"]
            detected_count = tut.get("readability_error_count", 0)
            if detected_count == 0 and tut.get("readability_suggestions"):
                detected_count = len(tut.get("readability_suggestions"))
        
        total_detected += detected_count
        
        file_stats.append({
            "file": file_path,
            "injected": injected_count,
            "detected": detected_count,
            "rate": (detected_count / injected_count * 100) if injected_count else 0
        })

    # Generate Markdown Report
    report_lines = []
    report_lines.append(f"# Benchmark Analysis Report\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Source**: {results_path.parent}\n")
    report_lines.append("\n---\n")
    
    report_lines.append("## Overall Performance\n")
    report_lines.append(f"- **Total Files**: {total_files}\n")
    report_lines.append(f"- **Total Errors Injected**: {total_injected}\n")
    report_lines.append(f"- **Total Errors Detected**: {total_detected}\n")
    overall_rate = (total_detected / total_injected * 100) if total_injected else 0
    report_lines.append(f"- **Overall Detection Rate**: {overall_rate:.1f}%\n")
    
    report_lines.append("\n## File-level Performance\n")
    report_lines.append("| File | Injected | Detected | Rate |\n")
    report_lines.append("|------|----------|----------|------|\n")
    
    for stat in sorted(file_stats, key=lambda x: x['rate'], reverse=True):
        report_lines.append(f"| {os.path.basename(stat['file'])} | {stat['injected']} | {stat['detected']} | {stat['rate']:.1f}% |\n")
        
    report_path = results_path.parent / "BENCHMARK_SUMMARY.md"
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
        
    print(f"Summary report saved to {report_path}")

def analyze_manifest_only(manifest_path):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"Analyzing injection manifest from {manifest_path.parent}")
    
    total_files = len(manifest.get("files", {}))
    total_injected = manifest.get("total_errors", 0)
    
    # Count by category
    categories = {}
    function_errors = 0
    
    for file_info in manifest.get("files", {}).values():
        for err in file_info.get("errors", []):
            cat = err.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            if cat == "function":
                function_errors += 1

    # Generate Markdown Report
    report_lines = []
    report_lines.append(f"# Injection Analysis Report (Partial)\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Source**: {manifest_path.parent}\n")
    report_lines.append("\n---\n")
    
    report_lines.append("## Injection Summary\n")
    report_lines.append(f"- **Total Files Targeted**: {total_files}\n")
    report_lines.append(f"- **Total Errors Injected**: {total_injected}\n")
    report_lines.append(f"- **Function Errors**: {function_errors}\n")
    
    report_lines.append("\n## Errors by Category\n")
    report_lines.append("| Category | Count |\n")
    report_lines.append("|----------|-------|\n")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"| {cat} | {count} |\n")
        
    report_path = manifest_path.parent / "INJECTION_SUMMARY.md"
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
        
    print(f"Injection summary saved to {report_path}")

if __name__ == "__main__":
    # Find latest output dir
    base_dir = Path("outputs/_tmp_satijalab_seurat_low")
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    if dirs:
        latest_dir = dirs[-1]
        results_path = latest_dir / "GEN_TEST_RESULTS.json"
        if results_path.exists():
            analyze_benchmark(results_path)
        elif (latest_dir / "INJECTION_MANIFEST.json").exists():
            print("Results file not found, analyzing manifest only...")
            analyze_manifest_only(latest_dir / "INJECTION_MANIFEST.json")
        else:
            print(f"Results file not found in {latest_dir}")
    else:
        print("No output directory found")
