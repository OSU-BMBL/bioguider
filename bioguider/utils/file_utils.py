import os
from enum import Enum
import json
# from adalflow.utils import get_adalflow_default_root_path
from pathlib import Path
from typing import Union, List, Optional, Tuple

import os
import string

try:
    import magic  # optional: pip install python-magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

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

def parse_repo_url(url: str) -> Tuple[Optional[str], Optional[str]]:
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

def parse_refined_repo_path(refined_repo_path: str) -> Tuple[Optional[str], Optional[str]]:
    repo_path = refined_repo_path.split("/")[-1]
    arr = repo_path.split("_")
    repo_name = arr[-1] if len(arr) > 1 else repo_path
    author = arr[0] if len(arr) > 1 else ""
    return author, repo_name

def retrieve_data_root_path() -> Path:
    data_folder = os.environ.get("DATA_FOLDER", "./data")
    root_folder = Path(data_folder, ".adalflow")
    return root_folder.absolute()
    
def detect_file_type(filepath, blocksize=2048, use_magic=True):
    """
    Detect if a file is text or binary.

    Args:
        filepath (str): Path to file.
        blocksize (int): Number of bytes to read for inspection.
        use_magic (bool): Use python-magic if available.

    Returns:
        str: "text" or "binary"
    """
    # Option 1: Use python-magic if available and requested
    if use_magic and HAS_MAGIC:
        try:
            mime = magic.from_file(filepath, mime=True)
            if mime and mime.startswith("text/"):
                return "text"
            return "binary"
        except Exception:
            pass  # fallback to heuristic

    # Option 2: Heuristic detection
    with open(filepath, "rb") as f:
        chunk = f.read(blocksize)
        if not chunk:  # empty file → treat as text
            return "text"

        # Null byte check
        if b"\0" in chunk:
            return "binary"

        # Check ratio of non-printable characters
        text_chars = bytearray(string.printable, "ascii")
        nontext = chunk.translate(None, text_chars)
        if float(len(nontext)) / len(chunk) > 0.30:
            return "binary"

    return "text"

def flatten_files(repo_path: Union[str, Path], files: Optional[List[str]]) -> List[str]:
    """
    Flatten directories into individual files.
    
    Args:
        repo_path (Union[str, Path]): The root path of the repository
        files (Optional[List[str]]): List of file/directory paths to flatten
        
    Returns:
        List[str]: List of individual file paths (directories are expanded to their contents)
    """
    if files is None:
        return []
    
    flattened = []
    repo_path = Path(repo_path)
    
    for file_path in files:
        full_path = repo_path / file_path
        
        if full_path.is_dir():
            # If it's a directory, recursively get all files in it
            for item in full_path.rglob("*"):
                if item.is_file():
                    # Get relative path from repo_path
                    rel_path = item.relative_to(repo_path)
                    flattened.append(str(rel_path))
        elif full_path.is_file():
            # If it's already a file, just add it
            flattened.append(file_path)
        # Skip if path doesn't exist
    
    return flattened

