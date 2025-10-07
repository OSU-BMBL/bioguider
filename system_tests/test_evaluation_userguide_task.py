import os
import shutil
import pytest
import logging

from bioguider.agents.evaluation_userguide_task import ConsistencyEvaluationResult, EvaluationUserGuideTask
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.managers.evaluation_manager import EvaluationManager
from bioguider.utils.code_structure_builder import CodeStructureBuilder
from bioguider.utils.utils import convert_to_serializable

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module", autouse=True)
def cleanup_after_tests(data_folder):
    """Cleanup function that runs after all tests in this module complete."""
    # This will run before any tests in this module
    yield  # This is where the tests run
            
    # Clean up database files
    db_path = os.path.join(data_folder, "databases")
    if os.path.exists(db_path):
        print(f"Cleaning up database directory: {db_path}")
        try:
            shutil.rmtree(db_path)
            print("✓ Database directory cleaned up")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up database directory: {e}")
    

@pytest.mark.skip(reason="Skipping this test")
def test_EvaluationInstallationTask_RepoAgent(llm, step_callback, root_path):
    files = ["README_CN.md", "requirements.txt", "markdown_docs/repo_agent/change_detector.md"]

    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
    )
    # evaluations, token_usage, files = task._evaluate(files)
    files = task._collect_files()

    # assert evaluations is not None
    assert files is not None and len(files) > 0

@pytest.mark.skip(reason="Skipping this test")
def test_EvaluationUserGuideTask_CollectFiles(llm, step_callback, root_path):
    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
    )
    files = task._collect_files()
    assert files is not None and len(files) > 0

@pytest.mark.skip(reason="Skipping this test")
def test_EvaluationUserGuideTask_ConsistencyEvaluation(llm, step_callback, root_path, data_folder):
    files = [
        'markdown_docs/repo_agent/multi_task_dispatch.md', 
        'markdown_docs/repo_agent/file_handler.md', 
        'markdown_docs/repo_agent/doc_meta_info.md', 
        'markdown_docs/repo_agent/project_manager.md', 
        'markdown_docs/repo_agent/settings.md', 
        'markdown_docs/repo_agent/chat_engine.md', 
        'markdown_docs/repo_agent/change_detector.md', 
        'markdown_docs/repo_agent/log.md', 
        'markdown_docs/tests/test_structure_tree.md'
    ]

    code_structure_db = CodeStructureDb(
        author="test",
        repo_name="test",
        data_folder=data_folder
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    
    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
    )
    
    for file in files:
        res = task._evaluate_consistency(file)
        assert res is None or type(res) == ConsistencyEvaluationResult

@pytest.mark.skip(reason="Skipping this test")
def test_EvaluationUserGuideTask_ConsistencyEvaluation(llm, step_callback, root_path, data_folder):
    files = [
        # 'README.md', 
        'src/RNAhybrid-2.1.2/man/RNAeffective.1', 
        'src/RNAhybrid-2.1.2/man/RNAcalibrate.1', 
        'src/RNAhybrid-2.1.2/man/RNAhybrid.1', 
        'src/RNAhybrid-2.1.2/README',
    ]

    code_structure_db = CodeStructureDb(
        author="Y2C99",
        repo_name="MIRROR",
        data_folder=data_folder
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/MIRROR",
        gitignore_path=f"{root_path}/MIRROR/.gitignore",
        code_structure_db=code_structure_db,
    )
    # code_structure_builder.build_code_structure()
    
    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/MIRROR",
        gitignore_path=f"{root_path}/MIRROR/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
    )
    
    res = task._evaluate(files)
    assert res is not None

@pytest.mark.skip(reason="Skipping this test")
def test_EvaluationUserGuideTask_evaluate(llm, step_callback, root_path, data_folder):
    code_structure_db = CodeStructureDb(
        author="Y2C99",
        repo_name="MIRROR",
        data_folder=data_folder
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/MIRROR",
        gitignore_path=f"{root_path}/MIRROR/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()

    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/MIRROR",
        gitignore_path=f"{root_path}/MIRROR/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
    )
    evaluations, files = task.evaluate()
    assert evaluations is not None
    assert files is not None

@pytest.mark.skip(reason="Skipping this test")
def test_EvaluationUserGuideTask_ConsistencyEvaluation_on_RepoAgent(llm, step_callback, root_path, data_folder):
    files = [
        # "markdown_docs/repo_agent/multi_task_dispatch.md", 
        "markdown_docs/repo_agent/file_handler.md", 
        # "markdown_docs/repo_agent/doc_meta_info.md", 
        # "markdown_docs/display/book_tools/generate_summary_from_book.md", 
        "markdown_docs/display/book_tools/generate_repoagent_books.md",
    ]

    code_structure_db = CodeStructureDb(
        author="OpenBMB",
        repo_name="RepoAgent",
        data_folder=data_folder
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    
    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
    )
    
    res = task._evaluate(files)
    import json
    from pydantic import BaseModel
        
    # Convert the result to a serializable format
    # res is a tuple: (evaluations_dict, token_usage_dict, files_list)
    serializable_res = convert_to_serializable(res)
    output = json.dumps(serializable_res, indent=2)
    logger.info(f"Evaluation result: {output}")
    assert res is not None

@pytest.mark.skip(reason="Skipping this test")
def test_EvaluationUserGuideTask_evaluate_on_telescope(llm, step_callback, root_path, data_folder):
    files = ["README.md"]

    code_structure_db = CodeStructureDb(
        author="mlbendall",
        repo_name="telescope",
        data_folder=data_folder
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/telescope",
        gitignore_path=f"{root_path}/telescope/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()

    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/telescope",
        gitignore_path=f"{root_path}/telescope/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
    )
    evaluations, files = task.evaluate()
    assert evaluations is not None
    assert files is not None

def test_EvaluationUserGuideTask_evaluate_on_BioGSP(llm, step_callback, root_path, data_folder):
    files = [
        # "README.md", 
        # "man/simulate_multiscale.Rd", 
        # "man/igft.Rd", 
        # "man/simulate_stripe_patterns.Rd", 
        # "man/visualize_multiscale.Rd",  # error occurred here
        "man/BioGSP-package.Rd", 
        "man/runSGWT.Rd", 
        "man/sgwt_auto_scales.Rd", 
        "man/simulate_checkerboard.Rd", 
        "man/hello_sgwt.Rd", 
        "man/plot_FM.Rd", "man/cosine_similarity.Rd", "man/initSGWT.Rd", "man/visualize_sgwt_kernels.Rd", "man/cal_laplacian.Rd", "man/FastDecompositionLap.Rd", "man/gft.Rd", "man/demo_sgwt.Rd", "man/plot_sgwt_decomposition.Rd", "man/print.SGWT.Rd", "man/visualize_similarity_xy.Rd", "man/find_knee_point.Rd", "man/sgwt_forward.Rd", "man/sgwt_inverse.Rd", "man/visualize_checkerboard.Rd", "man/install_and_load.Rd", "man/runSpecGraph.Rd", "man/compare_kernel_families.Rd", "man/codex_toy_data.Rd", "man/compute_sgwt_filters.Rd", "man/sgwt_energy_analysis.Rd", "man/visualize_stripe_patterns.Rd", "man/sgwt_get_kernels.Rd", "man/runSGCC.Rd", "man/sgwt-globals.Rd", "inst/extdata/SCALING_FUNCTIONS_GUIDE.md"]

    code_structure_db = CodeStructureDb(
        author="BMEngineeR",
        repo_name="BioGSP",
        data_folder=data_folder
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/BioGSP",
        gitignore_path=f"{root_path}/BioGSP/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()

    task = EvaluationUserGuideTask(
        llm=llm,
        repo_path=f"{root_path}/BioGSP",
        gitignore_path=f"{root_path}/BioGSP/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
    )
    # evaluations, files = task.evaluate()
    evaluations, token_usage, files = task._evaluate(files)
    
    assert evaluations is not None
    assert files is not None


