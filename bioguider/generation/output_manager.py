from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import OutputArtifact, GenerationManifest, PlannedEdit


class OutputManager:
    def __init__(self, base_outputs_dir: Optional[str] = None):
        self.base_outputs_dir = base_outputs_dir or "outputs"

    def prepare_output_dir(self, repo_url_or_name: str) -> str:
        repo_name = self._extract_repo_name(repo_url_or_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.base_outputs_dir, f"{repo_name}", timestamp)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def get_latest_output_dir(self, repo_url_or_name: str) -> str:
        repo_name = self._extract_repo_name(repo_url_or_name)
        out_dir = Path(self.base_outputs_dir, f"{repo_name}")
        latest_tm = datetime.min
        if not out_dir.exists():
            return None
        for f in out_dir.iterdir():
            if not f.is_dir():
                continue
            tm = f.name.split("/")[-1]
            if not tm.isdigit():
                continue
            tm = datetime.strptime(tm, "%Y%m%d_%H%M%S")
            if tm > latest_tm:
                latest_tm = tm
                latest_dir = f.name
        
        return latest_dir

    def _extract_repo_name(self, url_or_name: str) -> str:
        name = url_or_name.rstrip("/")
        if "/" in name:
            name = name.split("/")[-1]
        name = name.replace(".git", "")
        return name

    def write_files(self, output_dir: str, files: Dict[str, str], diff_stats_by_file: Dict[str, dict] | None = None) -> List[OutputArtifact]:
        artifacts: List[OutputArtifact] = []
        for rel_path, content in files.items():
            dest = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as fobj:
                fobj.write(content)
            artifacts.append(OutputArtifact(
                dest_rel_path=rel_path,
                original_rel_path=rel_path,
                change_summary="revised document",
                diff_stats=(diff_stats_by_file or {}).get(rel_path, {})
            ))
        return artifacts

    def write_manifest(
        self,
        output_dir: str,
        manifest: GenerationManifest,
    ) -> str:
        dest = os.path.join(output_dir, "manifest.json")
        with open(dest, "w", encoding="utf-8") as fobj:
            json.dump(manifest.model_dump(), fobj, indent=2)
        return dest


