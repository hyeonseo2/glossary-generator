from dataclasses import dataclass


@dataclass
class Paragraph:
    idx: int
    text: str


@dataclass
class Sentence:
    idx: int
    text: str


@dataclass
class AlignmentPair:
    i: int
    j: int
    sim: float


@dataclass
class TermEvidence:
    en_term: str
    ko_term: str
    score: float
    source_id: str
    en_para: int
    en_sent: int
    ko_para: int
    ko_sent: int
    example_en: str
    example_ko: str
    embedding_similarity: float
    context_similarity: float
    translit_similarity: float
    position_score: float
    alignment_score: float
