import re
import spacy

from .types import Sentence
from .config import PipelineConfig


class EnglishTermExtractor:
    ARTICLES = {"a", "an", "the"}
    STOPWORDS = {
        "this", "that", "these", "those", "it", "its", "they", "them", "their", "we",
        "you", "your", "my", "our", "he", "she", "his", "hers", "here", "there",
        "who", "what", "which", "one", "some", "most", "all", "any", "many",
        "every", "everyone", "everything", "anything",
    }

    def __init__(self, cfg: PipelineConfig, spacy_model: str):
        self.cfg = cfg
        self.nlp = spacy.load(spacy_model)

    def _strip_leading_articles(self, text: str) -> str:
        parts = text.split()
        if not parts:
            return text
        while parts and parts[0] in self.ARTICLES:
            parts = parts[1:]
        return " ".join(parts)

    def normalize(self, text: str) -> str:
        t = text.strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = t.strip(" ,.;:!?[](){}\"'")
        t = self._strip_leading_articles(t)
        return t

    def _is_stopword_phrase(self, chunk: str) -> bool:
        toks = [t for t in chunk.split() if t]
        if not toks:
            return True
        # Remove pure stopword chunks
        if all(t in self.STOPWORDS for t in toks):
            return True
        # Drop very short and ambiguous fragments
        if len(toks) == 1 and toks[0] in self.STOPWORDS:
            return True
        # Drop pronoun-led fragments like "your model", "this model"
        if toks[0] in self.STOPWORDS:
            return True
        return False

    def _valid_tokens(self, chunk: str) -> bool:
        n = len([t for t in chunk.split() if t])
        return self.cfg.min_en_tokens <= n <= self.cfg.max_en_tokens

    def _is_generic(self, chunk: str) -> bool:
        if chunk in self.cfg.en_generic_terms:
            return True
        if chunk in self.STOPWORDS:
            return True
        if re.fullmatch(r"\d+(\.\d+)?", chunk):
            return True
        return False

    def extract(self, sent: Sentence):
        doc = self.nlp(sent.text)
        out = []
        for nc in doc.noun_chunks:
            phrase = self.normalize(nc.text)
            if not phrase:
                continue
            if any(tok.pos_ == "VERB" for tok in nc):
                continue
            if not self._valid_tokens(phrase):
                continue
            if self._is_generic(phrase) or self._is_stopword_phrase(phrase):
                continue
            if not any(ch.isalpha() for ch in phrase):
                continue
            out.append(phrase)

        seen = set()
        return [x for x in out if not (x in seen or seen.add(x))]
