from verifier.config import Config
from verifier.textbook import (
    Passage,
    build_passages,
    chapter_summary,
    clean_pages,
    length_stats,
    load_passages,
    save_passages,
    split_sentences,
)

from conftest import MINI_TEXTBOOK


def test_split_sentences_keeps_abbreviations_together():
    text = "Keys, e.g. a primary key, identify tuples. Fig. 2 shows this. A new sentence follows."
    assert split_sentences(text) == [
        "Keys, e.g. a primary key, identify tuples.",
        "Fig. 2 shows this.",
        "A new sentence follows.",
    ]


def test_clean_pages_removes_page_numbers_and_repeated_headers_and_rejoins_hyphens():
    pages = [
        (1, "Database System Concepts\nThe body has a hyphen-\nated word.\n1"),
        (2, "Database System Concepts\nMore body text here.\n2"),
        (3, "Database System Concepts\nEven more body text.\n3"),
    ]
    cleaned = dict(clean_pages(pages))
    assert "hyphenated" in cleaned[1]
    assert not cleaned[1].rstrip().endswith("1")


def test_passages_follow_the_target_length_and_carry_locators():
    passages, info = build_passages(MINI_TEXTBOOK, Config(backend="stub"))
    assert len(passages) >= 3
    ids = [p.passage_id for p in passages]
    assert len(ids) == len(set(ids)), "passage ids must be unique"
    cfg = Config()
    for p in passages:
        assert p.chapter, "every passage needs a chapter"
        assert p.section, "every passage needs a section"
        assert p.page_start >= 1 and p.page_end >= p.page_start
        assert p.word_count == len(p.text.split())
        assert p.word_count <= cfg.passage_max_words + 40  # a final sentence may run over
    stats = length_stats(passages, cfg)
    assert stats["passages"] == len(passages)
    assert info["labeling_mode"] in {"headings", "outline"}


def test_a_passage_never_mixes_chapters():
    passages, _ = build_passages(MINI_TEXTBOOK, Config(backend="stub"))
    chapters = [row["chapter"] for row in chapter_summary(passages)]
    assert len(chapters) == len(set(chapters)) >= 3
    # passages come out in book order, so chapters appear as consecutive runs
    seen, last = [], None
    for p in passages:
        if p.chapter != last:
            seen.append(p.chapter)
            last = p.chapter
    assert len(seen) == len(set(seen))


def test_locator_format():
    one = Passage("p1", "text", "Chapter 1 A", "1.1 B", 4, 4, 1)
    two = Passage("p2", "text", "Chapter 1 A", None, 4, 5, 1)
    assert one.locator() == "Chapter 1 A; 1.1 B; p. 4"
    assert two.locator() == "Chapter 1 A; pp. 4-5"


def test_passages_round_trip(tmp_path):
    passages, _ = build_passages(MINI_TEXTBOOK, Config(backend="stub"))
    save_passages(passages, tmp_path / "p.jsonl")
    assert load_passages(tmp_path / "p.jsonl") == passages
