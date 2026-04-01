import csv
import os
from typing import Dict, List

from .config import PipelineConfig


class TransliterationScorer:
    def __init__(self, cfg: PipelineConfig):
        self.map = self._load_map(cfg.transliteration_map_path)

    def _load_map(self, path: str) -> Dict[str, List[str]]:
        if not os.path.exists(path):
            return {}
        mp: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en = (row.get("en") or row.get("token") or "").strip().lower()
                ko = (row.get("ko") or row.get("korean") or "").strip()
                if en and ko:
                    mp.setdefault(en, []).append(ko)
        return mp

    def score(self, en_term: str, ko_term: str) -> float:
        en = en_term.lower().strip()
        ko = ko_term.strip().lower()
        if not en or not ko:
            return 0.0

        tokens = [t for t in en.split() if t]
        if not tokens:
            return 0.0

        hit = 0.0
        for t in tokens:
            # exact dictionary mapping
            mapped = [m.lower() for m in self.map.get(t, [])]
            if any(m in ko for m in mapped):
                hit += 1.0
                continue
            if t in ko or ko in t:
                hit += 0.5

        return min(1.0, hit / len(tokens))
