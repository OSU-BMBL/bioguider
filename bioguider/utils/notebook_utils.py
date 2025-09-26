from __future__ import annotations
from pathlib import Path
from typing import Union, Dict, Any, List
import json

def extract_markdown_from_notebook(
    ipynb_path: Union[str, Path],
    out_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Extract markdown from a Jupyter notebook.
    """
    ipynb_path = Path(ipynb_path)
    if not ipynb_path.exists():
        raise FileNotFoundError(f"File {ipynb_path} does not exist")
    try:
        with ipynb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"File {ipynb_path} is not a valid JSON file")

    markdown_txts = [
        "\n".join(cell.get("source")) if isinstance(cell.get("source"), list) else cell.get("source") for cell in nb.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]
    text = "\n".join(markdown_txts)
    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text

def strip_notebook_to_code_and_markdown(
    ipynb_path: Union[str, Path],
    out_path: Union[str, Path, None] = None,
    keep_top_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Load a .ipynb and return a new notebook that:
      - keeps ONLY 'code' and 'markdown' cells
      - empties outputs and execution_count for code cells
      - drops all other cell types (e.g., 'raw')
      - preserves attachments on markdown cells
      - optionally preserves top-level metadata (kernelspec, language_info, etc.)

    Parameters
    ----------
    ipynb_path : str | Path
        Path to the input .ipynb file.
    out_path : str | Path | None, default None
        If provided, write the cleaned notebook to this path.
    keep_top_metadata : bool, default True
        If True, copy top-level metadata as-is (useful for re-running).
        If False, keep only minimal metadata.

    Returns
    -------
    dict
        The cleaned notebook (nbformat v4-style dict).
    """
    ipynb_path = Path(ipynb_path)
    if not ipynb_path.exists():
        raise FileNotFoundError(f"File {ipynb_path} does not exist")
    try:
        with ipynb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"File {ipynb_path} is not a valid JSON file")

    nbformat = nb.get("nbformat", 4)
    nbformat_minor = nb.get("nbformat_minor", 5)

    def _to_text(src) -> str:
        # nbformat allows str or list of lines
        if isinstance(src, list):
            return "".join(src)
        return src or ""

    new_cells: List[Dict[str, Any]] = []
    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            new_cell = {
                "cell_type": "markdown",
                "metadata": cell.get("metadata", {}),
                "source": _to_text(cell.get("source", "")),
            }
            if "attachments" in cell:
                new_cell["attachments"] = cell["attachments"]
            new_cells.append(new_cell)

        elif ctype == "code":
            new_cells.append({
                "cell_type": "code",
                "metadata": cell.get("metadata", {}),
                "source": _to_text(cell.get("source", "")),
                "execution_count": None,   # clear execution count
                "outputs": [],             # strip ALL outputs
            })

        # else: drop 'raw' and any other unknown cell types

    # Build new notebook object
    new_nb: Dict[str, Any] = {
        "nbformat": nbformat,
        "nbformat_minor": nbformat_minor,
        "metadata": nb.get("metadata", {}) if keep_top_metadata else {},
        "cells": new_cells,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(new_nb, f, ensure_ascii=False, indent=1)

    return new_nb

