import os
import pytest
import shutil
from pathlib import Path
from bioguider.managers.generation_test_manager_v2 import GenerationTestManagerV2
from bioguider.agents.agent_utils import write_file

def test_project_specific_injection(llm, step_callback):
    """
    Test that project-specific terms are extracted and injected as errors.
    """
    # Setup a dummy repo with some python code
    tmp_repo = "outputs/_tmp_test_project_injection"
    if os.path.exists(tmp_repo):
        shutil.rmtree(tmp_repo)
    os.makedirs(tmp_repo)
    
    # Create a dummy python file with specific function names
    code_content = """
def UniqueProjectFunction():
    pass

class SpecialProjectClass:
    def AnotherUniqueMethod(self):
        pass
"""
    os.makedirs(os.path.join(tmp_repo, "src"), exist_ok=True)
    write_file(os.path.join(tmp_repo, "src", "core.py"), code_content)
    
    # Create a dummy README that mentions these functions
    readme_content = """
# Test Project

This project uses UniqueProjectFunction() to do amazing things.
You can also use SpecialProjectClass to handle data.
Don't forget to call AnotherUniqueMethod() for processing.
"""
    write_file(os.path.join(tmp_repo, "README.md"), readme_content)
    
    mgr = GenerationTestManagerV2(llm, step_callback)
    
    # 1. Test extraction
    terms = mgr._extract_project_terms(tmp_repo)
    print(f"Extracted terms: {terms}")
    assert "UniqueProjectFunction" in terms
    assert "SpecialProjectClass" in terms
    assert "AnotherUniqueMethod" in terms
    
    # 2. Test injection
    target_files = {"readme": [os.path.join(tmp_repo, "README.md")]}
    manifests = mgr._inject_errors_into_files(target_files, tmp_repo, min_per_category=1)
    
    print(f"Manifest keys: {list(manifests.keys())}")
    
    all_errors = []
    for info in manifests.values():
        all_errors.extend(info.get("manifest", {}).get("errors", []))
    
    print("Injected errors:")
    found_project_error = False
    for err in all_errors:
        print(f"- {err['category']}: {err['original_snippet']} -> {err['mutated_snippet']}")
        if err['category'] == 'function' and ("UniqueProjectFunction" in err['original_snippet'] or "AnotherUniqueMethod" in err['original_snippet']):
            found_project_error = True
            
    assert found_project_error, "Should have injected an error into one of the project functions"
