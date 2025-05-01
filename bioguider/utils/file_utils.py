import os
from enum import Enum

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
