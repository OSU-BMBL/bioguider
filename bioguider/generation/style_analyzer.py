from __future__ import annotations

from typing import Dict

from .models import StyleProfile


class StyleAnalyzer:
    def analyze(self, files: Dict[str, str]) -> StyleProfile:
        profile = StyleProfile()

        # Infer heading style: prefer README
        readme = None
        for name in ("README.md", "README.rst"):
            if name in files:
                readme = files[name]
                break
        sample = readme or next(iter(files.values()), "")
        if "\n# " in sample or sample.startswith("# "):
            profile.heading_style = "#"
        elif "\n## " in sample:
            profile.heading_style = "#"
        else:
            profile.heading_style = "#"

        # List style
        if "\n- " in sample:
            profile.list_style = "-"
        elif "\n* " in sample:
            profile.list_style = "*"

        # Tone markers (heuristic): keep minimal
        profile.tone_markers = ["concise", "neutral"]
        return profile


