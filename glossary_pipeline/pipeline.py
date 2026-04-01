from typing import Callable, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from .config import PipelineConfig
from .preprocessing import Preprocessor
from .alignment import MonotonicAligner
from .term_extraction import EnglishTermExtractor
from .korean_candidates import KoreanCandidateGenerator
from .transliteration import TransliterationScorer
from .scoring import Scorer
from .voting import CorpusVoter


ProgressCallback = Callable[[str, int, int, str], None]


class GlossaryPipeline:
    def __init__(self, cfg: PipelineConfig = None, embed_model=None):
        self.cfg = cfg or PipelineConfig()
        self.embedder = embed_model or SentenceTransformer(self.cfg.embedding_model_name)

        self.pre = Preprocessor()
        self.aligner = MonotonicAligner(self.embedder, self.cfg)
        self.en_extractor = EnglishTermExtractor(self.cfg, self.cfg.spacy_model)
        self.ko_gen = KoreanCandidateGenerator(self.cfg)
        self.translit = TransliterationScorer(self.cfg)
        self.scorer = Scorer(self.embedder, self.cfg, self.translit)
        self.voter = CorpusVoter(self.cfg)

        self._evidence = []
        self._sources = set()

    def _run_pair(self, source_id: str, en_text: str, ko_text: str):
        en_paras = self.pre.split_paragraphs(en_text)
        ko_paras = self.pre.split_paragraphs(ko_text)

        para_pairs = self.aligner.align_paragraphs(en_paras, ko_paras)

        for ap in para_pairs:
            en_para = en_paras[ap.i]
            ko_para = ko_paras[ap.j]

            en_sents = self.pre.split_sentences(en_para.text)
            ko_sents = self.pre.split_sentences(ko_para.text)
            if not en_sents or not ko_sents:
                continue

            sent_pairs = self.aligner.align_sentences(en_sents, ko_sents)
            for sp in sent_pairs:
                en_sent = en_sents[sp.i]
                ko_sent = ko_sents[sp.j]

                en_terms = self.en_extractor.extract(en_sent)
                if not en_terms:
                    continue

                ko_cands = self.ko_gen.generate(ko_sent)
                if not ko_cands:
                    continue

                for en_term in en_terms[:self.cfg.topk]:
                    for ko_term in ko_cands[: self.cfg.topk]:
                        ev = self.scorer.score_pair(
                            en_term=en_term,
                            ko_term=ko_term,
                            en_sent=en_sent.text,
                            ko_sent=ko_sent.text,
                            en_para_i=ap.i,
                            en_sent_i=sp.i,
                            en_para_count=len(en_paras),
                            en_sent_count=len(en_sents),
                            ko_para_j=ap.j,
                            ko_sent_j=sp.j,
                            ko_para_count=len(ko_paras),
                            ko_sent_count=len(ko_sents),
                            para_sim=ap.sim,
                            sent_sim=sp.sim,
                            source_id=source_id,
                        )
                        self._evidence.append(ev)

    def fit_transform(
        self,
        pairs: List[Tuple[str, str, str]],
        progress_cb: Optional[ProgressCallback] = None,
        start_progress: int = 40,
        end_progress: int = 85,
    ):
        self._evidence = []
        self._sources = {pid for pid, _, _ in pairs}

        total = max(1, len(pairs))
        for idx, (source_id, en_text, ko_text) in enumerate(pairs, start=1):
            if progress_cb:
                progress_cb("run_pair_start", idx, total, source_id)

            self._run_pair(source_id, en_text, ko_text)

            if progress_cb:
                progress_cb("run_pair_done", idx, total, source_id)

        if progress_cb:
            progress_cb("aggregate", total, total, "정렬/투표 집계")

        return self.voter.aggregate(self._evidence, source_count=len(self._sources))

    def export(self, rows: List[dict], out_json: str = None, out_csv: str = None):
        self.voter.to_outputs(rows, out_json=out_json, out_csv=out_csv)
