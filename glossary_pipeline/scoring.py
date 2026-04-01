from typing import Dict

import numpy as np

from .types import TermEvidence
from .config import PipelineConfig


class Scorer:
    def __init__(self, emb_model, cfg: PipelineConfig, translit_scorer):
        self.model = emb_model
        self.cfg = cfg
        self.translit_scorer = translit_scorer
        self._embed_cache: Dict[str, np.ndarray] = {}

    def _embed(self, text: str):
        if text in self._embed_cache:
            return self._embed_cache[text]
        vec = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        self._embed_cache[text] = vec
        return vec

    def _sim(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        va = self._embed(a)
        vb = self._embed(b)
        s = float((va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        # map cosine [-1,1] to [0,1]
        return float((s + 1.0) / 2.0)

    def _position(self, en_sent_i: int, ko_sent_j: int, en_sent_total: int, ko_sent_total: int) -> float:
        if en_sent_total <= 1 or ko_sent_total <= 1:
            return 1.0
        e = en_sent_i / max(1, en_sent_total - 1)
        k = ko_sent_j / max(1, ko_sent_total - 1)
        return float(max(0.0, 1.0 - abs(e - k)))

    def score_pair(
        self,
        en_term: str,
        ko_term: str,
        en_sent: str,
        ko_sent: str,
        en_para_i: int,
        en_sent_i: int,
        en_para_count: int,
        en_sent_count: int,
        ko_para_j: int,
        ko_sent_j: int,
        ko_para_count: int,
        ko_sent_count: int,
        para_sim: float,
        sent_sim: float,
        source_id: str,
    ) -> TermEvidence:
        emb = self._sim(en_term, ko_term)
        ctx = self._sim(en_sent, ko_sent)
        translit = self.translit_scorer.score(en_term, ko_term)
        pos = self._position(en_sent_i, ko_sent_j, en_sent_count, ko_sent_count)
        align = 0.6 * para_sim + 0.4 * sent_sim

        score = (
            self.cfg.w_emb * emb
            + self.cfg.w_ctx * ctx
            + self.cfg.w_translit * translit
            + self.cfg.w_pos * pos
            + self.cfg.w_align * align
        )

        return TermEvidence(
            en_term=en_term,
            ko_term=ko_term,
            score=float(score),
            source_id=source_id,
            en_para=en_para_i,
            en_sent=en_sent_i,
            ko_para=ko_para_j,
            ko_sent=ko_sent_j,
            example_en=en_sent,
            example_ko=ko_sent,
            embedding_similarity=emb,
            context_similarity=ctx,
            translit_similarity=translit,
            position_score=pos,
            alignment_score=align,
        )
