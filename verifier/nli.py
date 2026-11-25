"""NLI scoring with a pretrained cross-encoder (thesis Section 3.2.4).

The passage is the premise and the claim is the hypothesis. The model is used
as released (no fine-tuning). Label order is read from the model's own
``id2label`` mapping instead of being assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .config import Config
from .retrieve import tokenize


@dataclass
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float


class NLIModel(Protocol):
    name: str

    def score(self, premises: list[str], hypothesis: str) -> list[NLIScores]:
        ...


class CrossEncoderNLI:
    """cross-encoder/nli-deberta-v3-base, loaded through transformers."""

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("pip install torch transformers to use the real backend") from exc
        self.name = model_name
        self._torch = torch
        self._max_length = max_length
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self._device).eval()
        id2label = {int(i): str(label).lower() for i, label in self._model.config.id2label.items()}
        self._index = {}
        for i, label in id2label.items():
            if "entail" in label:
                self._index["entailment"] = i
            elif "contradict" in label:
                self._index["contradiction"] = i
            elif "neutral" in label:
                self._index["neutral"] = i
        if set(self._index) != {"entailment", "contradiction", "neutral"}:
            raise RuntimeError(f"unexpected NLI labels in {model_name}: {id2label}")

    def score(self, premises: list[str], hypothesis: str) -> list[NLIScores]:
        torch = self._torch
        encoded = self._tokenizer(
            premises,
            [hypothesis] * len(premises),
            truncation=True,
            max_length=self._max_length,
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(**encoded).logits, dim=-1).cpu().numpy()
        return [
            NLIScores(
                entailment=float(row[self._index["entailment"]]),
                contradiction=float(row[self._index["contradiction"]]),
                neutral=float(row[self._index["neutral"]]),
            )
            for row in probabilities
        ]


_NEGATIONS = {"not", "no", "never", "cannot", "without", "neither", "nor"}
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on", "to", "and", "or",
    "that", "which", "it", "its", "as", "by", "for", "with", "at", "from", "this", "can", "has", "have",
}


def _is_negated(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return any(w in _NEGATIONS or w.endswith("n't") for w in words)


class StubNLI:
    """Lexical-overlap heuristic. STUB: checks the wiring only and says nothing about
    the accuracy of the real model. Never report its output as a result.

    For each passage it finds the sentence that shares the most content words with the claim,
    then compares whether that sentence and the claim are negated."""

    name = "stub-lexical-overlap"

    def score(self, premises: list[str], hypothesis: str) -> list[NLIScores]:
        claim_tokens = {t for t in tokenize(hypothesis) if t not in _STOP and t not in _NEGATIONS}
        claim_negated = _is_negated(hypothesis)
        results: list[NLIScores] = []
        for premise in premises:
            best_overlap, best_negated = 0.0, False
            for sentence in re.split(r"(?<=[.!?])\s+", premise):
                tokens = set(tokenize(sentence))
                overlap = len(claim_tokens & tokens) / len(claim_tokens) if claim_tokens else 0.0
                if overlap > best_overlap:
                    best_overlap, best_negated = overlap, _is_negated(sentence)
            if best_overlap >= 0.6 and claim_negated != best_negated:
                contradiction, entailment = 0.5 + 0.5 * best_overlap, 0.05
            elif best_overlap >= 0.6:
                entailment, contradiction = 0.5 + 0.5 * best_overlap, 0.05
            else:
                entailment, contradiction = 0.1 * best_overlap, 0.05 * best_overlap
            results.append(NLIScores(entailment, contradiction, max(0.0, 1.0 - entailment - contradiction)))
        return results


def make_nli(cfg: Config) -> NLIModel:
    if cfg.backend == "stub":
        return StubNLI()
    return CrossEncoderNLI(cfg.nli_model)
