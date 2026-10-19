import csv
import os
from typing import Dict, List, Tuple, Set

class AnnotationLoader:
    """Load SWL‑LSE label CSVs (train/val/test) and perform consistency checks.

    The CSVs have no header and contain two columns:
        recording_id (string) , class_id (int)
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.splits = ["train", "val", "test"]
        self.files = {split: os.path.join(base_dir, f"{split}_labels.csv") for split in self.splits}
        self.data: Dict[str, List[Tuple[str, int]]] = {}
        self.stats: Dict[str, Dict[str, int]] = {}
        self.all_ids: Set[str] = set()
        self.all_classes: Set[int] = set()

    def _load_csv(self, path: str) -> List[Tuple[str, int]]:
        rows: List[Tuple[str, int]] = []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for line_no, row in enumerate(reader, start=1):
                if len(row) != 2:
                    raise ValueError(f"{path}: line {line_no} does not have exactly 2 columns: {row}")
                rec_id, class_str = row[0].strip(), row[1].strip()
                if not rec_id:
                    raise ValueError(f"{path}: line {line_no} has empty recording_id")
                try:
                    class_id = int(class_str)
                except ValueError as e:
                    raise ValueError(f"{path}: line {line_no} class_id not integer: {class_str}") from e
                rows.append((rec_id, class_id))
        return rows

    def load(self) -> None:
        for split, path in self.files.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing CSV for split '{split}': {path}")
            rows = self._load_csv(path)
            self.data[split] = rows
            rec_ids = {r for r, _ in rows}
            class_ids = {c for _, c in rows}
            self.stats[split] = {
                "rows": len(rows),
                "unique_recordings": len(rec_ids),
                "unique_classes": len(class_ids),
                "duplicate_recordings": len(rows) - len(rec_ids),
            }
            overlap = self.all_ids.intersection(rec_ids)
            if overlap:
                raise ValueError(f"Recording ID overlap found between splits: {overlap}")
            self.all_ids.update(rec_ids)
            self.all_classes.update(class_ids)

    def summary(self) -> Dict[str, Dict[str, int]]:
        return self.stats

    def class_coverage(self) -> Tuple[int, int]:
        return len(self.all_classes), 300

    def validate(self) -> None:
        self.load()
        if any(c < 0 for c in self.all_classes):
            raise ValueError("Negative class_id encountered")
        for split, st in self.stats.items():
            if st["rows"] == 0:
                raise ValueError(f"Split '{split}' contains no rows")
        found, expected = self.class_coverage()
        if found != expected:
            raise ValueError(f"Class coverage mismatch: found {found}, expected {expected}")
