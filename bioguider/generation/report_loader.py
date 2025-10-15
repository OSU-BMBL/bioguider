from __future__ import annotations

import json
from typing import Tuple, Dict, Any

from .models import EvaluationReport


class EvaluationReportLoader:
    def _parse_bool(self, token: str) -> Any:
        if token == "True":
            return True
        if token == "False":
            return False
        return token

    def _split_args(self, s: str) -> Dict[str, Any]:
        # Split a function-like argument list into a dict, respecting quotes
        args: Dict[str, Any] = {}
        current = ""
        parts = []
        in_single = False
        in_double = False
        for ch in s:
            if ch == "'" and not in_double:
                in_single = not in_single
                current += ch
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                current += ch
                continue
            if ch == "," and not in_single and not in_double:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())
        for p in parts:
            if not p:
                continue
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1]
            else:
                # try bool/int
                if v in ("True", "False"):
                    v = self._parse_bool(v)
                else:
                    try:
                        v = int(v)
                    except Exception:
                        pass
            args[k] = v
        return args

    def _parse_structured_block(self, text: str, key: str) -> Dict[str, Any] | None:
        # Extract key=ClassName(arg1=val1, ...) and parse args
        marker = f"{key}="
        idx = text.find(marker)
        if idx == -1:
            return None
        rest = text[idx + len(marker) :]
        # find first '('
        pidx = rest.find("(")
        if pidx == -1:
            return None
        rest = rest[pidx + 1 :]
        # find matching ')'
        depth = 1
        collected = ""
        for ch in rest:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            collected += ch
        if not collected:
            return None
        return self._split_args(collected)

    def _parse_submission_eval_str(self, text: str) -> Dict[str, Any]:
        # Parse space-separated key=value pairs
        out: Dict[str, Any] = {}
        for token in text.strip().split():
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            v = v.strip()
            if v in ("True", "False"):
                out[k] = True if v == "True" else False
            else:
                out[k] = v
        return out

    def load(self, report_path: str) -> Tuple[EvaluationReport, str]:
        with open(report_path, "r", encoding="utf-8") as fobj:
            raw = json.load(fobj)

        # Normalize nested stringified fields if any
        def normalize(obj):
            if isinstance(obj, str):
                s = obj.strip()
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    try:
                        return json.loads(s)
                    except Exception:
                        return obj
                return obj
            if isinstance(obj, dict):
                return {k: normalize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [normalize(v) for v in obj]
            return obj

        normalized = normalize(raw)

        # Special handling for stringified evaluation fields
        inst_eval = normalized.get("installation")
        if isinstance(inst_eval, str):
            normalized["installation_evaluation"] = {
                "structured_evaluation": self._parse_structured_block(inst_eval["evaluation"], "structured_evaluation"),
            }
        else:
            normalized["installation_evaluation"] = inst_eval["evaluation"]
            normalized["installation_files"] = inst_eval["files"]

        readme_eval = normalized.get("readme")
        if isinstance(readme_eval["evaluations"], dict):
            fixed: Dict[str, Any] = {}
            for fname, val in readme_eval.items():
                if isinstance(val, str):
                    fixed[fname] = {
                        "structured_evaluation": self._parse_structured_block(val, "structured_evaluation"),
                    }
                else:
                    fixed[fname] = val
            normalized["readme_evaluation"] = fixed
            normalized["readme_files"] = readme_eval["files"]

        userguide_eval = normalized.get("userguide")
        if isinstance(userguide_eval["evaluation"], dict):
            normalized["userguide_evaluation"] = userguide_eval["evaluation"]
            normalized["userguide_files"] = userguide_eval["files"]

        # Tutorial evaluation handling
        tutorial_eval = normalized.get("tutorial")
        if tutorial_eval and isinstance(tutorial_eval.get("evaluation"), dict):
            normalized["tutorial_evaluation"] = tutorial_eval["evaluation"]
            normalized["tutorial_files"] = tutorial_eval["files"]

        # userguide_eval = normalized.get("userguide")
        # if isinstance(userguide_eval, str):
        #     normalized["userguide_evaluation"] = self._parse_structured_block(userguide_eval["evaluation"], "structured_evaluation")

        report = EvaluationReport(**normalized)
        return report, report_path


