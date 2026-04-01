from dataclasses import dataclass, field
from typing import Set


@dataclass
class PipelineConfig:
    # Alignment
    para_align_threshold: float = 0.35
    sent_align_threshold: float = 0.28
    para_align_window: int = 4
    sent_align_window: int = 6
    alignment_gap_penalty: float = 0.08

    # Term extraction
    max_en_tokens: int = 4
    min_en_tokens: int = 1
    en_generic_terms: Set[str] = field(
        default_factory=lambda: {
            "system",
            "software",
            "process",
            "method",
            "result",
            "analysis",
            "case",
            "data",
            "case study",
            "function",
            "feature",
            "feature set",
            "value",
            "time",
            "space",
            "type",
            "step",
            "steps",
            "field",
            "work",
            "workflow",
            "example",
            "examples",
            "preprocess",
            "preprocessing",
            "task",
            "tasks",
            "methodology",
        }
    )

    # Korean n-gram
    max_ko_ngram: int = 4
    min_ko_ngram: int = 1

    # Scoring weights
    w_emb: float = 0.35
    w_ctx: float = 0.20
    w_translit: float = 0.20
    w_pos: float = 0.10
    w_align: float = 0.15

    # Canonical confidence
    canonical_w_score_mean: float = 0.40
    canonical_w_support: float = 0.35
    canonical_w_best: float = 0.25

    # Voting / evidence
    min_evidence: int = 1
    topk: int = 2000
    min_score_mean: float = 0.62
    min_final_confidence: float = 0.70

    # Quality tuning
    strict_filter: bool = False

    # Models
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    spacy_model: str = "en_core_web_sm"

    # Transliteration dictionary
    transliteration_map_path: str = "data/default_transliteration_map.csv"
