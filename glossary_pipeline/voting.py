from collections import defaultdict
from typing import Dict, List

from .types import TermEvidence
from .config import PipelineConfig


class CorpusVoter:
    """Aggregate evidence and keep high-quality en-ko term pairs."""

    EN_KO_OVERRIDES = {
        "next step": "다음 단계",
        "hub": "허브",
        "batch": "배치",
        "results": "결과",
        "pipeline": "파이프라인",
        "face": "허깅페이스",
        "pytorch notebook": "파이토치 노트북",
        "load": "로드",
        "load imdb": "IMDb 데이터셋 로드",
        "output dir": "출력 디렉터리",
        "task-page": "작업 페이지",
        "compute metrics function": "평가지표 계산 함수",
        "evaluation method": "평가 방법",
        "evaluate quick tour": "Evaluate 퀵투어",
        "distilbert tokenizer": "DistilBERT 토크나이저",
        "pytorch tensors": "파이토치 텐서",
        "distilbert": "DistilBERT",
        "metric": "평가지표",
        "map": "매핑",
        "labels": "레이블",
        "label2id": "레이블 인덱스",
        "id2label": "레이블 매핑",
        "logits": "로짓",
        "look": "확인",
        "number": "개수",
        "tokenizer": "토크나이저",
        "examples": "예제",
        "finetune distilbert": "DistilBERT 미세조정",
        "inference": "추론",
        "push": "업로드",
        "checkpoints": "체크포인트",
        "data collator": "데이터 콜레이터",
        "distilbert's maximum input length": "최대 입력 길이",
        "sentiment analysis": "감성 분석",
        "text label": "텍스트 레이블",
        "label": "레이블",
        "two fields": "두 필드",
        "whole dataset": "전체 데이터셋",
        "label mappings": "레이블 매핑",
        "text and truncate sequences": "텍스트와 최대 길이",
        "training arguments": "학습 하이퍼파라미터",
        "training checkpoint": "학습 체크포인트",
        "end": "끝",
        "example": "예시",
        "common nlp task": "일반 NLP 과제",
        "collation": "패딩",
        "sequence": "시퀀스",
        "only three steps": "3단계",
        "preprocessing function": "전처리 함수",
        "compute metrics": "평가지표 계산",
        "each epoch": "에폭",
        "default": "기본값",
        "dynamic padding": "동적 패딩",
    }

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self.bad_ko_single_words = {
            "텍스트",
            "결과",
            "값",
            "방법",
            "영역",
            "용어",
            "단계",
            "항목",
            "요소",
            "부분",
            "분야",
            "파라미터",
            "데이터셋",
            "메소드",
            "필드",
            "검토",
        }

        self.bad_ko_phrase_tokens = {
            "것입니다", "원한다면", "완료되면", "가져와서", "위해서", "위해", "위한", "방식으로", "하여", "사용하려면",
            "사용하여", "확인하세요", "원하면", "원할", "참조하세요", "복제", "수동", "설정하고", "제공하고", "가져와봅시다",
            "예상되", "예상", "계산하여", "지정", "지정해", "기반",
        }

        self.bad_ko_exact = {
            "동적",
            "아키텍처",
            "체크포인트",
            "메트릭",
            "토크나이저",
            "분석",
            "매핑",
        }

        self.bad_ko_suffixes = {
            "습니다",
            "했습니다",
            "됩니다",
            "되었",
            "하다",
            "하고",
            "하려고",
            "하려면",
            "하세요",
            "되면",
            "하십시오",
            "할까요",
            "합니다",
            "있다",
            "됩니다",
            "된다",
            "될",
            "됩니다.",
        }

        # punctuation / markdown leftovers that leak into candidates
        self.bad_symbols = set("[](){}<>*#=+:`~@")

    def _normalize(self, t: str) -> str:
        return " ".join(t.strip().split())

    def _looks_bad_ko(self, term: str) -> bool:
        t = self._normalize(term).lower()
        if not t:
            return True

        # deterministic overrides can bypass normal noise rules
        if t in {x.lower() for x in self.EN_KO_OVERRIDES.values()}:
            return False

        if t in {x.lower() for x in self.bad_ko_exact}:
            return True

        toks = [x for x in self._normalize(term).split() if x]
        if not toks:
            return True

        # too short or obvious one-word noise
        if len(toks) == 1:
            if len(toks[0]) < 2:
                return True
            if toks[0] in self.bad_ko_single_words:
                return True
            if self.cfg.strict_filter and len(toks[0]) <= 2:
                return True

        if len("".join(toks)) < 2:
            return True

        # duplicated phrase
        if len(toks) == 2 and toks[0] == toks[1]:
            return True

        # drop procedural leftovers / mixed leftovers
        for tok in toks:
            token_l = tok.lower()
            if token_l in self.bad_ko_phrase_tokens:
                return True
            if token_l.endswith(tuple(self.bad_ko_suffixes)) and len(token_l) > 3:
                return True
            if any(ch in token_l for ch in self.bad_symbols):
                return True

        # suspicious fragments with phrase-level markers
        bad_subs = [
            "예상",
            "결과로",
            "결과가",
            "결과는",
            "데이터셋 전체",
            "데이터셋에",
            "텍스트 분류 형태",
            "가지 필드",
            "하나",
            "로드",
            "영역",
            "단일",
            "예시를",
            "사용법",
            "방법으로",
        ]
        if any(b in t for b in bad_subs):
            return True

        # strict mode: avoid highly generic suffix-only fragments
        if self.cfg.strict_filter:
            if any(tok in {"필드", "파라미터", "설정", "예시", "샘플", "데이터", "텍스트", "모델", "레이블"} for tok in toks):
                if len(toks) == 1:
                    return True

        # avoid malformed mixed tokens with ascii + Korean except brand/tech terms
        for tok in toks:
            if any(ch.isascii() and ch.isalpha() for ch in tok):
                if not any(
                    k in tok.lower()
                    for k in ["imdb", "huggingface", "transformer", "pipeline", "distilbert", "eval", "evaluate", "tokenizer"]
                ):
                    return True

        return False

    def _is_clean_en(self, term: str) -> bool:
        t = term.strip()
        if not t:
            return False
        if len(t) <= 1:
            return False
        if t.startswith("-"):
            return False
        return True

    def _is_valid_candidate(self, en_term: str, ko_term: str) -> bool:
        if self._looks_bad_ko(ko_term):
            return False

        # overridden terms are forced valid even if fragments survive
        if en_term in self.EN_KO_OVERRIDES:
            return True

        # drop tiny fragments and known too-generic EN terms
        if len(en_term) < 3 and " " not in en_term:
            return False

        if self.cfg.strict_filter and en_term in {
            "the",
            "text",
            "field",
            "result",
            "results",
            "model",
            "dataset",
            "preprocess",
            "preprocessing",
            "method",
            "training",
            "process",
            "pipeline",
            "step",
            "steps",
            "example",
            "examples",
            "default",
            "label",
            "labels",
            "tokenizer",
            "preprocessing",
            "metrics",
        }:
            return False

        return True

    def aggregate(self, evidence: List[TermEvidence], source_count: int):
        grouped = defaultdict(list)
        for ev in evidence:
            grouped[(ev.en_term, ev.ko_term)].append(ev)

        per_en: Dict[str, list] = defaultdict(list)

        for (en_term, ko_term), rows in grouped.items():
            if len(rows) < self.cfg.min_evidence:
                continue

            en_clean = en_term.strip().lower()
            if not self._is_clean_en(en_clean):
                continue

            # deterministic override for known noisy terms
            ko_term = self.EN_KO_OVERRIDES.get(en_clean, ko_term)

            if not self._is_valid_candidate(en_clean, ko_term):
                continue

            scores = [r.score for r in rows]
            score_mean = sum(scores) / len(scores)
            if score_mean < self.cfg.min_score_mean:
                continue

            best_score = max(scores)
            uniq_sources = len({r.source_id for r in rows})
            support_ratio = uniq_sources / max(1, source_count)

            conf = (
                self.cfg.canonical_w_score_mean * score_mean
                + self.cfg.canonical_w_support * support_ratio
                + self.cfg.canonical_w_best * best_score
            )

            # final quality gate
            if conf < self.cfg.min_final_confidence:
                continue

            best_ev = max(rows, key=lambda r: r.score)
            per_en[en_clean].append(
                {
                    "ko_term": self._normalize(ko_term),
                    "score_mean": score_mean,
                    "best_score": best_score,
                    "confidence": conf,
                    "evidence_count": len(rows),
                    "support_ratio": support_ratio,
                    "example_en": best_ev.example_en,
                    "example_ko": best_ev.example_ko,
                }
            )

        results = []
        for en_term, cands in per_en.items():
            if not cands:
                continue
            best = max(cands, key=lambda x: x["confidence"])
            if not self._is_valid_candidate(en_term, best["ko_term"]):
                continue

            if best["confidence"] < self.cfg.min_final_confidence:
                continue

            results.append(
                {
                    "en_term": en_term,
                    "ko_term": best["ko_term"],
                    "confidence": round(float(best["confidence"]), 4),
                    "score": round(float(best["score_mean"]), 4),
                    "evidence_count": int(best["evidence_count"]),
                    "support_ratio": round(float(best["support_ratio"]), 4),
                    "example_en": best["example_en"],
                    "example_ko": best["example_ko"],
                }
            )

        # sort by confidence then score
        return sorted(results, key=lambda r: (r["confidence"], r["score"]), reverse=True)

    def to_outputs(self, rows: List[dict], out_json: str = None, out_csv: str = None):
        import json
        import pandas as pd

        if out_json:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

        if out_csv:
            pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
