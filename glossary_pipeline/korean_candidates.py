import re
from typing import List, Set

from .types import Sentence
from .config import PipelineConfig


class KoreanCandidateGenerator:
    """Generate Korean phrase candidates via n-grams and remove basic Josa particles."""

    JOSA_SUFFIXES = {
        "은", "는", "이", "가", "을", "를", "에", "에서", "로", "으로", "와", "과", "에게", "께", "의", "도", "만", "부터", "까지", "보다", "로서", "임", "음", "으로써",
    }

    SENT_PUNCTS = set(".?!;:,…？！。！？\"'()[]{}=<>+-")
    HANGUL_RE = re.compile(r"[가-힣]")

    # Keep this short to avoid over-filtering valid noun phrases
    GARBAGE_TOKEN_SUFFIXES = {
        "하다", "합니다", "되어", "되었", "됩니다", "하려고", "하려면", "하세요", "수", "하기", "설정", "적용", "지정", "로드",
    }

    GARBAGE_TOKENS = {
        "수", "수있", "수있어", "있습니다", "있고", "있다", "입니다", "있을",
        "가장", "매우", "전체", "다양한", "실용적인", "동적", "긴",
        "많은", "모든", "여러", "최대", "긴", "작은", "큰", "어떤",
    }

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def _strip_particles(self, token: str) -> str:
        t = token.strip(".,()[]{}\"'")
        if not t:
            return t
        if t in self.JOSA_SUFFIXES:
            return ""

        for p in sorted(self.JOSA_SUFFIXES, key=len, reverse=True):
            if t.endswith(p) and len(t) > len(p):
                return t[: -len(p)]
        return t

    def _is_utterance_token(self, token: str) -> bool:
        """Filter tokens that are clearly not noun-like glossary terms."""
        if not self.HANGUL_RE.search(token):
            return False
        if token in self.GARBAGE_TOKENS:
            return True
        if len(token) <= 1:
            return True
        for s in self.GARBAGE_TOKEN_SUFFIXES:
            if token.endswith(s) and len(token) > len(s):
                # only filter when token is mostly verb phrase fragment
                return True
        return False

    def tokenize(self, text: str) -> List[str]:
        # keep sentence Korean tokens only
        text = re.sub(r"([가-힣])([A-Za-z])", r"\\1 \\2", text)
        text = re.sub(r"([A-Za-z])([가-힣])", r"\\1 \\2", text)
        text = re.sub(r"[\u200b]", " ", text)

        raw = re.split(r"\s+", re.sub(r"[\n\t]+", " ", text).strip())
        toks = []
        for t in raw:
            # keep Korean + basic alnum, then require korean presence
            t = re.sub(r"[^\w가-힣]+", "", t)
            if not t:
                continue
            if not self.HANGUL_RE.search(t):
                continue

            t2 = self._strip_particles(t)
            if not t2:
                continue
            if self._is_utterance_token(t2):
                continue
            toks.append(t2)
        return toks

    def _is_fragment(self, phrase: str) -> bool:
        if not phrase or len(phrase.replace(" ", "")) < 2:
            return True
        toks = phrase.split()
        # remove things like verb fragments in phrases
        for t in toks:
            if t in self.GARBAGE_TOKENS:
                return True
            for s in self.GARBAGE_TOKEN_SUFFIXES:
                if t.endswith(s) and len(t) > len(s):
                    return True
        if all(ch in self.SENT_PUNCTS for ch in phrase):
            return True
        return False

    def generate(self, sent: Sentence) -> List[str]:
        toks = self.tokenize(sent.text)
        if not toks:
            return []

        cands: Set[str] = set()
        N = len(toks)
        for n in range(self.cfg.min_ko_ngram, min(self.cfg.max_ko_ngram, N) + 1):
            for i in range(N - n + 1):
                phrase = " ".join(toks[i:i + n]).strip()
                if self._is_fragment(phrase):
                    continue
                cands.add(phrase)

        return sorted(cands, key=lambda x: (len(x), x))
