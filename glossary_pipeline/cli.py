import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import GlossaryPipeline


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run():
    ap = argparse.ArgumentParser(description="EN-KO bilingual glossary extraction")
    ap.add_argument("--input", required=True, help="JSON or JSONL with source_id, en_path, ko_path")
    ap.add_argument("--out-json", default="out/glossary.json")
    ap.add_argument("--out-csv", default="out/glossary.csv")
    ap.add_argument("--cfg", default=None, help="Optional JSON override config")
    ap.add_argument("--strict", action="store_true", help="Enable stricter quality filtering")
    ap.add_argument("--min-score-mean", type=float, default=None, help="Override minimum mean evidence score")
    ap.add_argument("--min-confidence", type=float, default=None, help="Override minimum final confidence")
    ap.add_argument("--pretty", action="store_true", help="Print a preview")
    args = ap.parse_args()

    # load document pairs
    pairs = []
    ext = Path(args.input).suffix.lower()
    if ext == ".jsonl":
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                pairs.append((row["source_id"], _read_text(row["en_path"]), _read_text(row["ko_path"])))
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            pairs.append((row["source_id"], _read_text(row["en_path"]), _read_text(row["ko_path"])))

    cfg = PipelineConfig()
    if args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            over = json.load(f)
            for k, v in over.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    if args.strict:
        cfg.strict_filter = True
    if args.min_score_mean is not None:
        cfg.min_score_mean = args.min_score_mean
    if args.min_confidence is not None:
        cfg.min_final_confidence = args.min_confidence

    pipe = GlossaryPipeline(cfg)
    rows = pipe.fit_transform(pairs)

    out_json = args.out_json
    out_csv = args.out_csv
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    pipe.export(rows, out_json=out_json, out_csv=out_csv)

    if args.pretty:
        for row in rows[:40]:
            print(row)
    else:
        print(f"Done. {len(rows)} canonical entries written.")
        print(f"JSON: {out_json}")
        print(f"CSV: {out_csv}")


if __name__ == "__main__":
    run()
