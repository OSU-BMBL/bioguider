import json
import sys

def inspect_function_errors(manifest_path):
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total files: {len(data.get('files', {}))}")
    print(f"Total errors: {data.get('total_errors', 0)}")
    
    function_errors = []
    for file_path, info in data.get('files', {}).items():
        for error in info.get('errors', []):
            if error.get('category') == 'function':
                function_errors.append({
                    'file': file_path,
                    'original': error.get('original_snippet'),
                    'mutated': error.get('mutated_snippet'),
                    'rationale': error.get('rationale')
                })
    
    print(f"\nFound {len(function_errors)} function errors:")
    for i, err in enumerate(function_errors[:20], 1):
        print(f"{i}. [{err['file']}] {err['original']} -> {err['mutated']}")

if __name__ == "__main__":
    inspect_function_errors("outputs/_tmp_satijalab_seurat_low/INJECTION_MANIFEST.json")
