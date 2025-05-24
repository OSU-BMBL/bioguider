
from bioguider.utils.file_utils import remove_output_cells, extract_code_from_notebook

def test_remove_output_cells():
    notebook_path = "/home/ubuntu/projects/github/Slide_recon/demo/example.ipynb"
    temp_content = remove_output_cells(notebook_path)
    with open("./temp.ipynb", "w") as fobj:
        fobj.write(temp_content)


def test_extract_code_from_notebook():
    notebook_path = "/home/ubuntu/projects/github/Slide_recon/demo/example.ipynb"
    code = extract_code_from_notebook(notebook_path)
    with open("./temp.py", "w") as fobj:
        fobj.write(code)