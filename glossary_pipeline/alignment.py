from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .types import Paragraph, Sentence, AlignmentPair
from .config import PipelineConfig


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
    return float(sim)


class MonotonicAligner:
    def __init__(self, emb_model, cfg: PipelineConfig):
        self.model = emb_model
        self.cfg = cfg

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        return np.asarray(self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True))

    def align_sequences(
        self,
        a_texts: List[str],
        b_texts: List[str],
        threshold: float,
        window: int,
        gap_penalty: float,
    ) -> List[AlignmentPair]:
        if not a_texts or not b_texts:
            return []

        emb_a = self._encode(a_texts)
        emb_b = self._encode(b_texts)
        n, m = len(a_texts), len(b_texts)

        # similarity matrix
        sims = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                sims[i, j] = _cosine(emb_a[i], emb_b[j])

        dp = np.full((n + 1, m + 1), -1e9, dtype=float)
        bt = np.full((n + 1, m + 1), "", dtype=object)
        dp[0, 0] = 0.0

        for i in range(1, n + 1):
            dp[i, 0] = dp[i - 1, 0] - gap_penalty
            bt[i, 0] = "skip_a"
        for j in range(1, m + 1):
            dp[0, j] = dp[0, j - 1] - gap_penalty
            bt[0, j] = "skip_b"

        for i in range(1, n + 1):
            j_start = max(1, i - window)
            j_end = min(m, i + window)
            for j in range(j_start, j_end + 1):
                match = dp[i - 1, j - 1] + sims[i - 1, j - 1]
                skip_a = dp[i - 1, j] - gap_penalty
                skip_b = dp[i, j - 1] - gap_penalty
                best = max(match, skip_a, skip_b)
                if best == match:
                    bt[i, j] = "match"
                elif best == skip_a:
                    bt[i, j] = "skip_a"
                else:
                    bt[i, j] = "skip_b"
                dp[i, j] = best

        pairs = []
        i, j = n, m
        while i > 0 and j > 0:
            op = bt[i, j]
            if op == "match":
                s = sims[i - 1, j - 1]
                if s >= threshold:
                    pairs.append(AlignmentPair(i - 1, j - 1, float(s)))
                i -= 1
                j -= 1
            elif op == "skip_a":
                i -= 1
            elif op == "skip_b":
                j -= 1
            else:
                break

        pairs.reverse()
        return pairs

    def align_paragraphs(self, en_paras: List[Paragraph], ko_paras: List[Paragraph]) -> List[AlignmentPair]:
        return self.align_sequences(
            [p.text for p in en_paras],
            [p.text for p in ko_paras],
            self.cfg.para_align_threshold,
            self.cfg.para_align_window,
            self.cfg.alignment_gap_penalty,
        )

    def align_sentences(self, en_sents: List[Sentence], ko_sents: List[Sentence]) -> List[AlignmentPair]:
        return self.align_sequences(
            [s.text for s in en_sents],
            [s.text for s in ko_sents],
            self.cfg.sent_align_threshold,
            self.cfg.sent_align_window,
            self.cfg.alignment_gap_penalty,
        )
