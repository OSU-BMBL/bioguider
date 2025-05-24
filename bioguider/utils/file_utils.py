import os
from enum import Enum
import json

class FileType(Enum):
    unknown = "u"
    file = "f"
    directory = "d"
    symlink = "l"
    broken_symlink = "broken symlink"

def get_file_type(file_path: str) -> FileType:
    """
    Get the file type of a given file path.
    
    Args:
        file_path (str): The path to the file or directory.
    
    Returns:
        FileType: The type of the file (file, directory, or symlink).
    """
    if os.path.isfile(file_path):
        return FileType.file
    elif os.path.isdir(file_path):
        return FileType.directory
    elif os.path.islink(file_path):
        try:
            os.stat(file_path)
            return FileType.symlink
        except FileNotFoundError:
            return FileType.broken_symlink
        except Exception:
            return FileType.unknown
    else:
        # raise ValueError(f"Unknown file type for path: {file_path}")
        return FileType.unknown

def remove_output_cells(notebook_path: str) -> str:
    """
    Remove output cells from a Jupyter notebook to reduce its size.

    Args:
        notebook_path (str): Path to the input Jupyter notebook file.
        output_path (str): Path to save the modified notebook file.
    """
    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        notebook = json.load(nb_file)

    notebook['cells'] = [
        cell for cell in notebook.get('cells', []) 
        if cell.get('cell_type') != 'markdown'
    ]
    for cell in notebook.get('cells'):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        

    return json.dumps(notebook)

def extract_code_from_notebook(notebook_path: str) -> str:
    """
    Extract all code from a Jupyter notebook.

    Args:
        notebook_path (str): Path to the input Jupyter notebook file.

    Returns:
        str: A concatenated string of all code cells.
    """
    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        notebook = json.load(nb_file)

    # Extract code from cells of type 'code'
    code_cells = [
        '\n'.join(cell['source']) for cell in notebook.get('cells', [])
        if cell.get('cell_type') == 'code'
    ]
    code_cells = [
        cell.replace("\n\n", "\n") for cell in code_cells
    ]

    # Combine all code cells into a single string
    return '\n\n'.join(code_cells)

