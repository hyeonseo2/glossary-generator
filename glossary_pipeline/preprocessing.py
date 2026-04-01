import re
from typing import List

from .types import Paragraph, Sentence


class Preprocessor:
    def clean_text(self, text: str) -> str:
        # remove comments
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

        # remove fenced code blocks first
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

        # remove inline code backticks
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # remove markdown links keeping only visible text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # remove bracket refs and inline refs like [Trainer], [~foo], [[foo]]
        text = re.sub(r"\[\[[^\]]+\]\]", " ", text)
        text = re.sub(r"\[[^\]]+\]", " ", text)

        # strip HTML tags like <Youtube .../>, <br>, etc.
        text = re.sub(r"</?[^>]+>", " ", text)

        # remove markdown heading marks
        text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", text)

        # remove doc REPL markers
        text = re.sub(r"(?m)^\s*>>>\s*", " ", text)
        text = re.sub(r"(?m)^\s*\.\.\.\s*", " ", text)

        # remove bullet prefixes and numbering artifacts
        text = re.sub(r"(?m)^\s*[-•*]\s+", " ", text)

        # remove emoji / pictograph / flag tokens
        text = re.sub(r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\u2600-\u27BF\U0001F1E6-\U0001F1FF]", " ", text)
        text = text.replace("🙂", " ").replace("🙁", " ").replace("😐", " ")

        # remove strong/emphasis and inline style markers
        text = text.replace("**", " ").replace("*", " ").replace("__", " ").replace("_", " ")

        # normalize whitespace/newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\u200b", "", text)
        text = re.sub(r"\xa0", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_paragraphs(self, text: str) -> List[Paragraph]:
        text = self.clean_text(text)
        raw = re.split(r"\n{2,}", text)
        paras = []
        idx = 0
        for p in raw:
            p = p.strip()
            if not p:
                continue
            paras.append(Paragraph(idx, p))
            idx += 1
        return paras

    def split_sentences(self, text: str) -> List[Sentence]:
        # Lightweight multilingual sentence split
        units = re.split(r"(?<=[\.!\?。！？])\s+|\n+", text.strip())
        out = []
        idx = 0
        for u in units:
            u = u.strip()
            if not u or len(u) <= 1:
                continue
            out.append(Sentence(idx, u))
            idx += 1
        return out
