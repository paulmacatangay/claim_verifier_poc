"""Decision rule (thesis Eq. 3.11).

With E the highest entailment probability and C the highest contradiction
probability over the top-k retrieved passages:

    Supported             if E >= tau_E and E > C
    Contradicted          if C >= tau_C and C > E
    Insufficient Evidence otherwise
"""

from __future__ import annotations

from dataclasses import dataclass

from .nli import NLIScores

SUPPORTED = "Supported"
CONTRADICTED = "Contradicted"
INSUFFICIENT = "Insufficient Evidence"
LABELS = (SUPPORTED, CONTRADICTED, INSUFFICIENT)


@dataclass
class Decision:
    label: str
    entailment: float            # E
    contradiction: float         # C
    best_entail_index: int       # index (0-based) of the passage that gave E
    best_contra_index: int       # index (0-based) of the passage that gave C
    reason: str

    @property
    def review_flag(self) -> bool:
        """Contradicted and Insufficient Evidence claims go to instructor review."""
        return self.label != SUPPORTED


def decide(scores: list[NLIScores], tau_entail: float, tau_contra: float) -> Decision:
    if not scores:
        raise ValueError("at least one NLI score is required")
    best_e = max(range(len(scores)), key=lambda i: scores[i].entailment)
    best_c = max(range(len(scores)), key=lambda i: scores[i].contradiction)
    e = scores[best_e].entailment
    c = scores[best_c].contradiction

    if e >= tau_entail and e > c:
        label = SUPPORTED
        reason = f"E={e:.2f} reached tau_E={tau_entail:.2f} and exceeds C={c:.2f}."
    elif c >= tau_contra and c > e:
        label = CONTRADICTED
        reason = f"C={c:.2f} reached tau_C={tau_contra:.2f} and exceeds E={e:.2f}."
    else:
        label = INSUFFICIENT
        reason = (
            f"Neither rule fired: E={e:.2f} (tau_E={tau_entail:.2f}), C={c:.2f} (tau_C={tau_contra:.2f}). "
            "The retrieved passages do not establish or refute the claim."
        )
    return Decision(label, e, c, best_e, best_c, reason)
