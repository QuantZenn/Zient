import re
import json
from pathlib import Path
from typing import Dict, List, Union, Optional


class FormatRegistry:
    def __init__(self):
        self.registry_path = Path(__file__).resolve().parents[0] / "formats.json"
        self.formats: List[Dict] = self._load_formats()

    def _load_formats(self) -> List[Dict]:
        if not self.registry_path.exists():
            return []
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_formats(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.formats, f, indent=2)

    def match_sample(self, sample_text: str) -> Optional[Dict]:
        """
        Try to match a text sample against regex-based format rules.
        """
        for format_def in self.formats:
            rule = format_def.get("match_rule", {})
            if rule.get("type") == "regex":
                pattern = rule.get("pattern")
                if pattern and re.search(pattern, sample_text):
                    return format_def
        return None

    def match(self, sample: Union[str, Dict]) -> Optional[Dict]:
        """
        Match a sample (either text or column dict) against registered formats.
        """
        if isinstance(sample, str):
            # Try regex match on string sample
            match = self.match_sample(sample)
            if match:
                return match

        if isinstance(sample, dict):
            for format_def in self.formats:
                rule = format_def.get("match_rule", {})
                rule_type = rule.get("type")

                if rule_type in {"column", "column_match"}:
                    required_columns = rule.get("columns", [])
                    if all(col in sample for col in required_columns):
                        return format_def

        return None

    def register(self, name: str, description: str, match_rule: Dict, extractor: str, overwrite: bool = False):
        """
        Add or update a format definition in the registry.
        """
        existing = next((fmt for fmt in self.formats if fmt["name"] == name), None)
        new_format = {
            "name": name,
            "description": description,
            "match_rule": match_rule,
            "extractor": extractor
        }

        if existing:
            if overwrite:
                self.formats = [new_format if fmt["name"] == name else fmt for fmt in self.formats]
            else:
                return  # Skip silently
        else:
            self.formats.append(new_format)

        self._save_formats()

    def list_formats(self) -> List[Dict]:
        return self.formats
