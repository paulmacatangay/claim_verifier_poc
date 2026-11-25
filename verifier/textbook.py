"""Textbook ingestion and segmentation (thesis Section 3.2.2).

The prescribed textbook is read from a PDF (page numbers and outline are used) or
from a plain-text file whose pages are separated by form feeds. Text is cut into
passages of about 100 to 150 words. Every passage keeps its chapter, section and
page so that each verdict can point back to the book.

The licensed textbook is never copied into the audit logs; only passage
identifiers are logged. Keep the textbook file and the index built from it out
of any shared repository.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Iterator

from .config import Config


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------
@dataclass
class Passage:
    passage_id: str
    text: str
    chapter: str | None
    section: str | None
    page_start: int
    page_end: int
    word_count: int

    def locator(self) -> str:
        """Human-readable location, for example 'Chapter 3 ...; 3.2 ...; p. 87'."""
        parts = [p for p in (self.chapter, self.section) if p]
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return "; ".join(parts + [pages])


@dataclass
class Block:
    text: str
    page: int
    chapter: str | None
    section: str | None


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _open_pdf(path: Path):
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:
        try:
            import fitz  # older PyMuPDF
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("PyMuPDF is required to read PDFs: pip install pymupdf") from exc
    return fitz.open(str(path))


def read_pages(path: str | Path, cfg: Config) -> tuple[list[tuple[int, str]], list[tuple[int, str, int]]]:
    """Return ([(printed_page, text)], [(outline_level, title, printed_page)])."""
    path = Path(path)
    suffix = path.suffix.lower()
    pages: list[tuple[int, str]] = []
    toc: list[tuple[int, str, int]] = []
    if suffix == ".pdf":
        doc = _open_pdf(path)
        try:
            for index, page in enumerate(doc, start=1):
                if index < cfg.first_pdf_page:
                    continue
                pages.append((index + cfg.page_offset, page.get_text("text")))
            for level, title, page_no in doc.get_toc(simple=True):
                if page_no >= 1:
                    toc.append((int(level), str(title).strip(), int(page_no) + cfg.page_offset))
        finally:
            doc.close()
    elif suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        for index, text in enumerate(raw.split("\f"), start=1):
            if index < cfg.first_pdf_page:
                continue
            pages.append((index + cfg.page_offset, text))
    else:
        raise ValueError(f"unsupported textbook format {suffix!r}; use .pdf or .txt")
    return pages, toc


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop page numbers and running headers or footers, and rejoin hyphenated words."""
    n_pages = len(pages)

    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter[str] = Counter()
    for _, text in pages:
        counts.update({norm(l) for l in text.splitlines() if l.strip() and len(l.strip()) <= 80})
    threshold = max(3, int(0.3 * n_pages)) if n_pages >= 10 else None

    cleaned: list[tuple[int, str]] = []
    for page_no, text in pages:
        kept: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.fullmatch(r"\d{1,4}", stripped):
                continue
            if threshold and len(stripped) <= 80 and counts[norm(stripped)] >= threshold:
                continue
            kept.append(stripped)
        joined = "\n".join(kept)
        joined = re.sub(r"(\w)-\n([a-z])", r"\1\2", joined)
        cleaned.append((page_no, joined))
    return cleaned


# ---------------------------------------------------------------------------
# chapter and section labelling
# ---------------------------------------------------------------------------
CHAPTER_RE = re.compile(r"^\s*chapter\s+(\d+)\b[\s:.\-–—]*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+([A-Z][^.!?]{2,90})$")


def _flatten(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _blocks_from_toc(
    pages: list[tuple[int, str]], toc: list[tuple[int, str, int]], cfg: Config
) -> list[Block]:
    entries = sorted(
        (e for e in toc if e[0] in (cfg.toc_chapter_level, cfg.toc_section_level)),
        key=lambda e: e[2],
    )
    blocks: list[Block] = []
    for page_no, text in pages:
        chapter: str | None = None
        section: str | None = None
        for level, title, start in entries:
            if start > page_no:
                break
            if level == cfg.toc_chapter_level:
                chapter, section = title, None
            else:
                section = title
        flat = _flatten(text)
        if flat:
            blocks.append(Block(flat, page_no, chapter, section))
    return blocks


def _blocks_from_headings(pages: list[tuple[int, str]]) -> list[Block]:
    blocks: list[Block] = []
    chapter: str | None = None
    section: str | None = None
    for page_no, text in pages:
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                blocks.append(Block(" ".join(buffer), page_no, chapter, section))
                buffer.clear()

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m_chapter = CHAPTER_RE.match(stripped)
            if m_chapter:
                flush()
                title = m_chapter.group(2).strip()
                chapter = f"Chapter {m_chapter.group(1)}" + (f" {title}" if title else "")
                section = None
                continue
            m_section = SECTION_RE.match(stripped)
            if m_section:
                flush()
                section = f"{m_section.group(1)} {m_section.group(2).strip()}"
                continue
            buffer.append(stripped)
        flush()
    return blocks


def make_blocks(
    pages: list[tuple[int, str]], toc: list[tuple[int, str, int]], cfg: Config
) -> tuple[list[Block], str]:
    has_chapters = any(level == cfg.toc_chapter_level for level, _, _ in toc)
    if has_chapters:
        return _blocks_from_toc(pages, toc, cfg), "outline"
    return _blocks_from_headings(pages), "headings"


# ---------------------------------------------------------------------------
# sentence splitting and segmentation
# ---------------------------------------------------------------------------
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "fig.", "figs.", "eq.", "eqs.", "vs.", "cf.", "sec.", "no.", "al.")
_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[])')
_TERMINATORS = ('.', '!', '?', ':', '"', '”', ')')


def split_sentences(text: str) -> list[str]:
    parts = _SPLIT_RE.split(text.strip())
    sentences: list[str] = []
    for part in parts:
        if sentences:
            previous = sentences[-1].lower()
            if previous.endswith(_ABBREVIATIONS) or re.search(r"\b[A-Z]\.$", sentences[-1]):
                sentences[-1] = f"{sentences[-1]} {part}"
                continue
        sentences.append(part)
    return [s for s in sentences if s]


@dataclass
class _Sentence:
    text: str
    page: int
    chapter: str | None
    section: str | None


def _iter_sentences(blocks: Iterable[Block]) -> Iterator[_Sentence]:
    """Split blocks into sentences, rejoining a sentence that runs across a page break."""
    carry: _Sentence | None = None
    for block in blocks:
        pieces = split_sentences(block.text)
        if not pieces:
            continue
        if carry is not None:
            same_place = carry.chapter == block.chapter and carry.section == block.section
            if same_place:
                pieces[0] = f"{carry.text} {pieces[0]}"
                first_page = carry.page
            else:
                yield carry
                first_page = block.page
            carry = None
        else:
            first_page = block.page
        for i, piece in enumerate(pieces):
            page = first_page if i == 0 else block.page
            is_last = i == len(pieces) - 1
            if is_last and not piece.rstrip().endswith(_TERMINATORS):
                carry = _Sentence(piece, page, block.chapter, block.section)
            else:
                yield _Sentence(piece, page, block.chapter, block.section)
    if carry is not None:
        yield carry


def _hard_split(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def segment(blocks: Iterable[Block], cfg: Config) -> tuple[list[Passage], int]:
    """Cut blocks into passages. Returns (passages, number_of_dropped_fragments)."""
    passages: list[Passage] = []
    dropped = 0
    current: list[_Sentence] = []
    words = 0

    def flush() -> None:
        nonlocal current, words, dropped
        if not current:
            return
        if words < cfg.passage_drop_below_words:
            dropped += 1
        else:
            passages.append(
                Passage(
                    passage_id=f"P-{len(passages) + 1:05d}",
                    text=" ".join(s.text for s in current),
                    chapter=current[0].chapter,
                    section=current[0].section,
                    page_start=current[0].page,
                    page_end=current[-1].page,
                    word_count=words,
                )
            )
        current, words = [], 0

    for sentence in _iter_sentences(blocks):
        if current and (
            sentence.chapter != current[0].chapter or sentence.section != current[0].section
        ):
            flush()
        n_words = len(sentence.text.split())
        pieces = (
            _hard_split(sentence.text, cfg.passage_max_words)
            if n_words > cfg.passage_max_words
            else [sentence.text]
        )
        for piece in pieces:
            piece_words = len(piece.split())
            if current and words >= cfg.passage_min_words and words + piece_words > cfg.passage_max_words:
                flush()
            current.append(_Sentence(piece, sentence.page, sentence.chapter, sentence.section))
            words += piece_words
            if words >= cfg.passage_max_words:
                flush()
    flush()
    return passages, dropped


# ---------------------------------------------------------------------------
# build, save, load, summarise
# ---------------------------------------------------------------------------
def build_passages(path: str | Path, cfg: Config) -> tuple[list[Passage], dict]:
    pages, toc = read_pages(path, cfg)
    pages = clean_pages(pages)
    blocks, labeling = make_blocks(pages, toc, cfg)
    passages, dropped = segment(blocks, cfg)
    if not passages:
        raise ValueError("no passages were produced; check the file, first_pdf_page and page_offset")
    info = {
        "labeling_mode": labeling,
        "pages_read": len(pages),
        "outline_entries": len(toc),
        "dropped_fragments": dropped,
    }
    return passages, info


def chapter_summary(passages: list[Passage]) -> list[dict]:
    order: list[str | None] = []
    stats: dict[str | None, dict] = {}
    for p in passages:
        if p.chapter not in stats:
            order.append(p.chapter)
            stats[p.chapter] = {"chapter": p.chapter, "passages": 0, "first_page": p.page_start, "last_page": p.page_end}
        entry = stats[p.chapter]
        entry["passages"] += 1
        entry["first_page"] = min(entry["first_page"], p.page_start)
        entry["last_page"] = max(entry["last_page"], p.page_end)
    return [stats[c] for c in order]


def length_stats(passages: list[Passage], cfg: Config) -> dict:
    counts = [p.word_count for p in passages]
    in_range = sum(cfg.passage_min_words <= c <= cfg.passage_max_words for c in counts)
    return {
        "passages": len(counts),
        "min_words": min(counts),
        "median_words": median(counts),
        "max_words": max(counts),
        "share_within_target": round(in_range / len(counts), 3),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(path: str | Path, cfg: Config, passages: list[Passage], info: dict, encoder_name: str) -> dict:
    """Corpus manifest for the source-lock stage (Table 3.5, stage 1)."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "textbook_id": cfg.textbook_id,
        "edition": cfg.edition,
        "source_file": Path(path).name,
        "source_sha256": file_sha256(path),
        "segmentation": {
            "passage_min_words": cfg.passage_min_words,
            "passage_max_words": cfg.passage_max_words,
            "passage_drop_below_words": cfg.passage_drop_below_words,
            "first_pdf_page": cfg.first_pdf_page,
            "page_offset": cfg.page_offset,
        },
        "reading": info,
        "length_stats": length_stats(passages, cfg),
        "chapters": chapter_summary(passages),
        "encoder": encoder_name,
    }


def save_passages(passages: list[Passage], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")


def load_passages(path: str | Path) -> list[Passage]:
    with open(path, encoding="utf-8") as handle:
        return [Passage(**json.loads(line)) for line in handle if line.strip()]
