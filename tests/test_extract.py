import pytest

from verifier.extract import ClaimExtractor


@pytest.fixture(scope="module")
def extractor(spacy_model):
    return ClaimExtractor(spacy_model)


def claims(extractor, text):
    return [c.text for c in extractor.extract(text).claims]


def test_questions_and_instructions_are_skipped_with_a_reason(extractor):
    result = extractor.extract("What is a primary key? Define a primary key.")
    assert result.claims == []
    assert {s.reason for s in result.skipped} == {"question", "instruction, not a factual statement"}


def test_negation_is_never_dropped(extractor):
    assert claims(extractor, "A view does not store data.") == ["A view does not store data."]


def test_shared_negation_is_not_split(extractor):
    text = "A view does not store data and has no index."
    assert claims(extractor, text) == [text]


def test_ellipsis_is_not_split(extractor):
    text = "A view stores no data, but a materialized view does."
    assert claims(extractor, text) == [text]


def test_compound_sentence_is_split_into_atomic_claims(extractor):
    out = claims(extractor, "A primary key uniquely identifies each tuple, and a relation has only one primary key.")
    assert out == ["A primary key uniquely identifies each tuple.", "A relation has only one primary key."]


def test_semicolon_splits_and_pronoun_inside_a_sentence_is_resolved(extractor):
    out = claims(extractor, "Atomicity is enforced by the concurrency control system; durability is guaranteed by logging.")
    assert out == ["Atomicity is enforced by the concurrency control system.", "Durability is guaranteed by logging."]
    result = extractor.extract("Normalization reduces redundancy but it can increase the number of joins.")
    assert [c.text for c in result.claims] == [
        "Normalization reduces redundancy.",
        "Normalization can increase the number of joins.",
    ]
    assert result.claims[0].notes == [] and result.claims[1].notes  # the note sits on the claim it concerns


def test_pronoun_from_previous_sentence_is_resolved_but_flagged(extractor):
    result = extractor.extract("A relation has a primary key. It must be unique.")
    second = result.claims[1]
    assert second.text == "A relation must be unique."
    assert any("confirm" in w for w in second.warnings)
    assert result.claims[0].warnings == []


def test_unresolvable_pronoun_is_kept_with_a_warning(extractor):
    result = extractor.extract("It is simple.")
    assert [c.text for c in result.claims] == ["It is simple."]
    assert any("unresolved reference" in w for w in result.claims[0].warnings)


def test_headings_and_fragments_are_skipped(extractor):
    assert extractor.extract("Chapter 3").claims == []


def test_source_offsets_point_back_into_the_submitted_text(extractor):
    text = "A view is a virtual relation.   Indexes speed up lookups by key."
    for claim in extractor.extract(text).claims:
        assert text[claim.source_start : claim.source_end] == claim.sentence


def test_duplicate_claims_are_reported_once(extractor):
    out = claims(extractor, "A view is a virtual relation. A view is a virtual relation.")
    assert out == ["A view is a virtual relation."]


def test_missing_model_gives_a_clear_error():
    with pytest.raises(RuntimeError, match="spacy download"):
        ClaimExtractor("no_such_model_xyz")
