"""Eq. 3.11: S if E >= tau_E and E > C; C if C >= tau_C and C > E; otherwise I.
E and C are the maxima over the retrieved passages."""

import pytest

from verifier.decision import CONTRADICTED, INSUFFICIENT, SUPPORTED, decide
from verifier.nli import NLIScores


def s(e, c, n=None):
    return NLIScores(e, c, 1.0 - e - c if n is None else n)


TAU = 0.80


@pytest.mark.parametrize(
    "scores, label",
    [
        ([s(0.95, 0.02)], SUPPORTED),
        ([s(0.80, 0.10)], SUPPORTED),                       # E equal to the threshold counts
        ([s(0.799, 0.10)], INSUFFICIENT),                   # just below the threshold
        ([s(0.05, 0.95)], CONTRADICTED),
        ([s(0.10, 0.80)], CONTRADICTED),
        ([s(0.10, 0.799)], INSUFFICIENT),
        ([s(0.10, 0.10, 0.80)], INSUFFICIENT),              # neutral dominates
        ([s(0.90, 0.90, 0.0)], INSUFFICIENT),               # tie: neither is strictly larger
        ([s(0.95, 0.85, 0.0)], SUPPORTED),                  # both reach tau, E is larger
        ([s(0.85, 0.95, 0.0)], CONTRADICTED),               # both reach tau, C is larger
    ],
)
def test_truth_table(scores, label):
    assert decide(scores, TAU, TAU).label == label


def test_uses_maximum_over_passages_not_one_passage():
    scores = [s(0.1, 0.1), s(0.92, 0.01), s(0.2, 0.2), s(0.05, 0.60), s(0.0, 0.0)]
    decision = decide(scores, TAU, TAU)
    assert decision.label == SUPPORTED
    assert decision.best_entail_index == 1
    assert decision.entailment == pytest.approx(0.92)
    assert decision.contradiction == pytest.approx(0.60)


def test_support_and_contradiction_in_different_passages():
    scores = [s(0.90, 0.0), s(0.0, 0.97)]   # E=0.90 in one passage, C=0.97 in another
    decision = decide(scores, TAU, TAU)
    assert decision.label == CONTRADICTED
    assert decision.best_contra_index == 1


def test_thresholds_are_separate():
    scores = [s(0.85, 0.0)]
    assert decide(scores, 0.90, 0.80).label == INSUFFICIENT
    assert decide(scores, 0.80, 0.90).label == SUPPORTED


def test_review_flag_for_everything_except_supported():
    assert decide([s(0.95, 0.0)], TAU, TAU).review_flag is False
    assert decide([s(0.0, 0.95)], TAU, TAU).review_flag is True
    assert decide([s(0.3, 0.3)], TAU, TAU).review_flag is True


def test_reason_is_recorded():
    assert "tau" in decide([s(0.3, 0.3)], TAU, TAU).reason


def test_no_scores_is_an_error():
    with pytest.raises(ValueError):
        decide([], TAU, TAU)
