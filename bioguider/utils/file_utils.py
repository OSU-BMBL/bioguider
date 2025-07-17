import os
from enum import Enum
import json
# from adalflow.utils import get_adalflow_default_root_path
from pathlib import Path

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

def parse_repo_url(url: str) -> tuple[str | None, str | None]:
    """
    Parses a git repository URL to extract the author/organization and repository name.

    Args:
        url: The repository URL (e.g., HTTPS or SSH).

    Returns:
        A tuple containing (author_or_org, repo_name), or (None, None) if parsing fails.
    """
    try:
        # Handle SSH format first (e.g., git@github.com:user/repo.git)
        if '@' in url and ':' in url:
            path_part = url.split(':')[-1]
        # Handle HTTPS format (e.g., https://github.com/user/repo.git)
        else:
            path_part = url.split('://')[-1].split('/', 1)[-1]

        # Clean up the path
        if path_part.endswith('.git'):
            path_part = path_part[:-4]

        parts = path_part.split('/')
        if len(parts) >= 2:
            author = parts[-2]
            repo_name = parts[-1]
            return author, repo_name
        else:
            return None, None
    except Exception:
        return None, None

def retrieve_data_root_path() -> Path:
    data_folder = os.environ.get("DATA_FOLDER", "./data")
    root_folder = Path(data_folder, ".adalflow")
    return root_folder.absolute()
    




