import json
import os
import shutil
import pytest
import logging

from bioguider.agents.evaluation_tutorial_task import EvaluationTutorialTask
from bioguider.agents.evaluation_userguide_task import ConsistencyEvaluationResult, EvaluationUserGuideTask
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.database.summarized_file_db import SummarizedFilesDb
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
def test_EvaluationTutorialTask_telescope(llm, step_callback, root_path, data_folder):
    code_structure_db = CodeStructureDb(
        author="mlbendall",
        repo_name="telescope",
        data_folder=data_folder
    )
    summarized_files_db = SummarizedFilesDb(
        author="mlbendall",
        repo_name="telescope",
        data_folder=data_folder,
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/telescope",
        gitignore_path=f"{root_path}/telescope/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    task = EvaluationTutorialTask(
        llm=llm,
        repo_path=f"{root_path}/telescope",
        gitignore_path=f"{root_path}/telescope/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
        summarized_files_db=summarized_files_db,
    )
    evaluations, files = task.evaluate()
    assert evaluations is not None
    assert files is not None

@pytest.mark.skip()
def test_EvaluationTutorialTask_RepoAgent(llm, step_callback, root_path, data_folder):
    code_structure_db = CodeStructureDb(
        author="OpenBMB",
        repo_name="RepoAgent",
        data_folder=data_folder
    )
    summarized_files_db = SummarizedFilesDb(
        author="OpenBMB",
        repo_name="RepoAgent",
        data_folder=data_folder,
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    task = EvaluationTutorialTask(
        llm=llm,
        repo_path=f"{root_path}/RepoAgent",
        gitignore_path=f"{root_path}/RepoAgent/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
        summarized_files_db=summarized_files_db,
    )
    evaluations, files = task.evaluate()
    assert evaluations is not None
    assert files is not None

@pytest.mark.skip()
def test_EvaluationTutorialTask_scanpy(llm, step_callback, root_path, data_folder):
    code_structure_db = CodeStructureDb(
        author="scverse",
        repo_name="scanpy",
        data_folder=data_folder
    )
    summarized_files_db = SummarizedFilesDb(
        author="scverse",
        repo_name="scanpy",
        data_folder=data_folder,
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/scanpy",
        gitignore_path=f"{root_path}/scanpy/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    task = EvaluationTutorialTask(
        llm=llm,
        repo_path=f"{root_path}/scanpy",
        gitignore_path=f"{root_path}/scanpy/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
        summarized_files_db=summarized_files_db,
    )
    evaluations, files = task.evaluate()
    assert evaluations is not None
    assert files is not None

@pytest.mark.skip()
def test_EvaluationTutorialTask_only_evaluate_on_scanpy(llm, step_callback, root_path, data_folder):
    code_structure_db = CodeStructureDb(
        author="scverse",
        repo_name="scanpy",
        data_folder=data_folder
    )
    summarized_files_db = SummarizedFilesDb(
        author="scverse",
        repo_name="scanpy",
        data_folder=data_folder,
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/scanpy",
        gitignore_path=f"{root_path}/scanpy/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    task = EvaluationTutorialTask(
        llm=llm,
        repo_path=f"{root_path}/scanpy",
        gitignore_path=f"{root_path}/scanpy/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
        summarized_files_db=summarized_files_db,
    )
    files = [
        "notebooks/dask.ipynb", 
        # "notebooks/basic-scrna-tutorial.ipynb", 
        # "notebooks/tutorial_pearson_residuals.ipynb", 
        # "notebooks/paga-paul15.ipynb", 
        # "notebooks/pbmc3k.ipynb", 
        # "notebooks/integrating-data-using-ingest.ipynb", 
        # "notebooks/how-to/plotting-with-marsilea.ipynb", 
        # "notebooks/how-to/cell-cycle.ipynb", 
        # "notebooks/how-to/knn-transformers.ipynb", 
        # "notebooks/scanpy_workshop/day2_02_GeneSetEnrichmentAnalysis.ipynb", 
        # "notebooks/scanpy_workshop/day1_01_GColab.ipynb", 
        # "notebooks/scanpy_workshop/day1_01_GColabs_solutions.ipynb", 
        # "notebooks/scanpy_workshop/day2_03_RNAvelocity.ipynb", 
        # "notebooks/scanpy_workshop/day2_02_GSEA_Colabs.ipynb", 
        # "notebooks/scanpy_workshop/day1_01.ipynb", 
        # "notebooks/scanpy_workshop/day1_01_solutions.ipynb", 
        # "notebooks/scanpy_workshop/day2_01_DifferentialExpression.ipynb", 
        # "notebooks/scanpy_workshop/day2_01_DE_Colabs.ipynb", 
        # "docs/tutorials/plotting/advanced.ipynb", 
        # "docs/tutorials/plotting/core.ipynb", 
        # "docs/tutorials/experimental/pearson_residuals.ipynb", 
        # "docs/tutorials/experimental/dask.ipynb", 
        # "docs/tutorials/basics/clustering-2017.ipynb", 
        # "docs/tutorials/basics/clustering.ipynb", 
        # "docs/tutorials/basics/integrating-data-using-ingest.ipynb", 
        # "docs/tutorials/trajectories/paga-paul15.ipynb"
    ]
    evaluations, files = task._evaluate(files)
    assert evaluations is not None
    assert files is not None

@pytest.mark.skip()
def test_EvaluationTutorialTask_on_seurat(llm, step_callback, root_path, data_folder):
    code_structure_db = CodeStructureDb(
        author="satijalab",
        repo_name="seurat",
        data_folder=data_folder
    )
    summarized_files_db = SummarizedFilesDb(
        author="satijalab",
        repo_name="seurat",
        data_folder=data_folder,
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/seurat",
        gitignore_path=f"{root_path}/seurat/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    task = EvaluationTutorialTask(
        llm=llm,
        repo_path=f"{root_path}/seurat",
        gitignore_path=f"{root_path}/seurat/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
        summarized_files_db=summarized_files_db,
    )
    evaluations, files = task.evaluate()
    assert evaluations is not None
    assert files is not None

def test_EvaluationTutorialTask_evaluate_on_seurat(llm, step_callback, root_path, data_folder):
    code_structure_db = CodeStructureDb(
        author="satijalab",
        repo_name="seurat",
        data_folder=data_folder
    )
    summarized_files_db = SummarizedFilesDb(
        author="satijalab",
        repo_name="seurat",
        data_folder=data_folder,
    )
    code_structure_builder = CodeStructureBuilder(
        repo_path=f"{root_path}/seurat",
        gitignore_path=f"{root_path}/seurat/.gitignore",
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    task = EvaluationTutorialTask(
        llm=llm,
        repo_path=f"{root_path}/seurat",
        gitignore_path=f"{root_path}/seurat/.gitignore",
        step_callback=step_callback,
        code_structure_db=code_structure_db,
        summarized_files_db=summarized_files_db,
    )
    files = [
        "vignettes/seurat5_multimodal_vignette.Rmd", 
        "vignettes/seurat5_v4_changes.Rmd", 
    ]
    evaluations, token_usage, files = task._evaluate(files)
    serializable_dict = convert_to_serializable(evaluations)
    with open("seurat_tutorial_evaluation.json", "w") as f:
        json.dump(serializable_dict, f, indent=2)

    assert evaluations is not None
    assert files is not None

    