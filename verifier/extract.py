"""Atomic claim extraction (thesis Section 3.2.2).

spaCy's dependency parser finds sentences and clauses; rule-based decomposition
splits coordinated main clauses into one claim each; decontextualisation replaces
an unambiguous pronoun subject with the noun phrase it refers to.

The rules are deliberately conservative. The thesis requires that entities,
values and negation are preserved, so a sentence is left whole whenever a split
could change its meaning (a shared negation, a shared leading adjunct, a clause
with no content of its own, or coordination other than "and" / "but"). A claim
that still depends on context is kept and carries a warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
_SPLIT_CONJUNCTIONS = {"and", "but"}
_SINGULAR_PRONOUNS = {"it", "this", "that"}
_PLURAL_PRONOUNS = {"they", "these", "those"}
_OBJECT_PRONOUNS = {"it", "they", "them", "this", "these", "those"}
_COMPLEMENT_DEPS = {"attr", "acomp", "dobj", "obj", "prep", "xcomp", "ccomp", "oprd", "agent", "dative"}
_LEADING_ADJUNCTS = {"prep", "advmod", "advcl", "npadvmod", "mark"}


@dataclass
class ExtractedClaim:
    text: str
    sentence: str
    source_start: int
    source_end: int
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class SkippedSentence:
    text: str
    reason: str
    source_start: int
    source_end: int


@dataclass
class ExtractionResult:
    claims: list[ExtractedClaim]
    skipped: list[SkippedSentence]


class ClaimExtractor:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("pip install spacy to extract claims") from exc
        try:
            self.nlp = spacy.load(model)
        except OSError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(f"spaCy model {model!r} is missing: python -m spacy download {model}") from exc
        self.model_name = model

    # ------------------------------------------------------------------
    def extract(self, text: str) -> ExtractionResult:
        doc = self.nlp(text)
        claims: list[ExtractedClaim] = []
        skipped: list[SkippedSentence] = []
        seen: set[str] = set()
        last_subject: tuple[str, str | None] | None = None  # (noun phrase, "sing" | "plur" | None)

        for sent in doc.sents:
            raw = sent.text
            sentence = raw.strip()
            if not sentence:
                continue
            start = sent.start_char + (len(raw) - len(raw.lstrip()))
            end = start + len(sentence)
            pieces = (
                [self.nlp(p.strip())[:] for p in sentence.split(";") if p.strip()]
                if ";" in sentence
                else [sent]
            )
            for span in pieces:
                reason, parser_warning = self._screen(span)
                if reason:
                    skipped.append(SkippedSentence(span.text.strip(), reason, start, end))
                    continue
                replacements, warnings, notes = self._resolve_references(span, last_subject)
                if parser_warning:
                    warnings.append((None, parser_warning))
                for clause, token_ids in self._split_clauses(span, replacements):
                    key = clause.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    claims.append(
                        ExtractedClaim(
                            clause,
                            sentence,
                            start,
                            end,
                            [m for i, m in warnings if i is None or i in token_ids],
                            [m for i, m in notes if i is None or i in token_ids],
                        )
                    )
                last_subject = self._subject_phrase(span) or last_subject
        return ExtractionResult(claims, skipped)

    # ------------------------------------------------------------------
    @staticmethod
    def _screen(span) -> tuple[str | None, str | None]:
        """Return (skip_reason, warning). A sentence is skipped only when it clearly is not a
        factual statement; when the parser is unsure the sentence is kept and flagged, so that
        no claim is silently lost."""
        text = span.text.strip()
        if text.endswith("?"):
            return "question", None
        if len([t for t in span if t.is_alpha]) < 3:
            return "too short to be a claim", None
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in span)
        has_subject = any(t.dep_ in _SUBJECTS for t in span)
        if has_verb and has_subject:
            return None, None
        if has_verb and span.root.tag_ == "VB":
            return "instruction, not a factual statement", None
        if text.endswith("."):
            return None, "the parser found no clear subject or verb; check this claim"
        return "no clear subject or verb (heading or fragment)", None

    # ------------------------------------------------------------------
    @staticmethod
    def _subject_phrase(span) -> tuple[str, str | None] | None:
        """Noun phrase that is the subject of the main clause, with its grammatical number."""
        for token in span.root.children:
            if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ in ("NOUN", "PROPN"):
                for chunk in span.doc.noun_chunks:
                    if chunk.root.i == token.i:
                        number = token.morph.get("Number")
                        kind = "sing" if "Sing" in number else "plur" if "Plur" in number else None
                        return chunk.text, kind
        return None

    @staticmethod
    def _has_local_antecedent(span, pronoun, wanted: str) -> bool:
        for token in span:
            if token.i >= pronoun.i:
                break
            if token.pos_ in ("NOUN", "PROPN"):
                number = token.morph.get("Number")
                if ("Sing" in number and wanted == "sing") or ("Plur" in number and wanted == "plur"):
                    return True
        return False

    def _resolve_references(self, span, last_subject):
        replacements: dict[int, str] = {}
        warnings: list[tuple[int | None, str]] = []  # (token index the message is about, message)
        notes: list[tuple[int | None, str]] = []
        sentence_subject = self._subject_phrase(span)
        for token in span:
            lower = token.lower_
            if token.pos_ == "PRON" and token.dep_ in ("nsubj", "nsubjpass") and (
                lower in _SINGULAR_PRONOUNS or lower in _PLURAL_PRONOUNS
            ):
                wanted = "sing" if lower in _SINGULAR_PRONOUNS else "plur"
                # a main-clause pronoun refers back to the previous sentence; a pronoun in a
                # coordinated clause refers to the subject of this sentence; anything else is left flagged
                from_previous = token.head.i == span.root.i
                if from_previous:
                    antecedent = last_subject
                elif token.head in span.root.conjuncts:
                    antecedent = sentence_subject
                else:
                    antecedent = None
                if antecedent and antecedent[1] == wanted:
                    phrase = antecedent[0]
                    if token.i != span.start and re.match(r"(The|A|An) ", phrase):
                        phrase = phrase[0].lower() + phrase[1:]
                    replacements[token.i] = phrase
                    notes.append((token.i, f"'{token.text}' was replaced by '{antecedent[0]}'"))
                    if from_previous:
                        # the previous subject is only a guess at the referent, so ask for a check
                        warnings.append(
                            (
                                token.i,
                                f"'{token.text}' was taken to mean the previous sentence's subject "
                                f"'{antecedent[0]}'; confirm that this is the intended reference",
                            )
                        )
                elif antecedent is None and self._has_local_antecedent(span, token, wanted):
                    pass  # the pronoun refers to a noun earlier in the same sentence
                else:
                    warnings.append((token.i, f"unresolved reference: '{token.text}'"))
            elif token.pos_ == "PRON" and lower in _OBJECT_PRONOUNS and token.dep_ not in _SUBJECTS:
                warnings.append((token.i, f"pronoun '{token.text}' may need context"))
            elif token.dep_ == "det" and lower in {"this", "these", "those"} and token.i == span.start:
                warnings.append((token.i, f"context-dependent reference: '{token.text} {token.head.text}'"))
        return replacements, warnings, notes

    # ------------------------------------------------------------------
    @staticmethod
    def _render(tokens, replacements: dict[int, str]) -> str:
        pieces = []
        for token in sorted(tokens, key=lambda t: t.i):
            pieces.append(replacements.get(token.i, token.text) + token.whitespace_)
        text = "".join(pieces)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:?!)])", r"\1", text)
        text = text.strip(" ,;:")
        if not text:
            return ""
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
        return text

    def _split_clauses(self, span, replacements: dict[int, str]) -> list[tuple[str, set[int]]]:
        """Return (claim text, indices of the tokens it was built from) for each clause."""
        whole = [(self._render(list(span), replacements), {t.i for t in span})]
        root = span.root
        verbs = [root] + [
            t for t in root.conjuncts if t.pos_ in ("VERB", "AUX") and t.i != root.i and span.start <= t.i < span.end
        ]
        if len(verbs) < 2:
            return whole
        coordinators = [t for t in span if t.dep_ == "cc" and t.head in verbs]
        if not coordinators or any(c.lower_ not in _SPLIT_CONJUNCTIONS for c in coordinators):
            return whole

        root_subject = next((c for c in root.children if c.dep_ in _SUBJECTS), None)
        if root_subject is None:
            return whole
        root_aux = [c for c in root.children if c.dep_ in ("aux", "auxpass")]
        root_negated = any(c.dep_ == "neg" for c in root.children)
        leading = [c for c in root.children if c.dep_ in _LEADING_ADJUNCTS and c.i < root_subject.i]

        for verb in verbs[1:]:
            children = list(verb.children)
            own_subject = any(c.dep_ in _SUBJECTS for c in children)
            own_aux = any(c.dep_ in ("aux", "auxpass") for c in children)
            if not own_subject and (root_negated and not own_aux):
                return whole  # negation may scope over both clauses
            if not own_subject and leading:
                return whole  # a leading adjunct may scope over both clauses
        for verb in verbs:
            if verb.lemma_ in {"be", "do", "have"}:
                if not any(c.dep_ in _COMPLEMENT_DEPS for c in verb.children):
                    return whole  # ellipsis, the clause has no content of its own

        owned: dict[int, set[int]] = {}
        for verb in verbs:
            tokens = {t.i for t in verb.subtree if span.start <= t.i < span.end}
            for other in verbs:
                if other.i != verb.i and other.i in tokens:
                    tokens -= {t.i for t in other.subtree}
            tokens -= {c.i for c in coordinators}
            owned[verb.i] = tokens

        clauses: list[tuple[str, set[int]]] = []
        for verb in verbs:
            tokens = [span.doc[i] for i in owned[verb.i]]
            if verb.i != root.i:
                own_subject = any(c.dep_ in _SUBJECTS for c in verb.children)
                own_aux = any(c.dep_ in ("aux", "auxpass") for c in verb.children)
                prefix: list = []
                if not own_subject:
                    prefix += [t for t in root_subject.subtree if span.start <= t.i < span.end]
                    if not own_aux:
                        prefix += root_aux
                tokens = prefix + tokens
            text = self._render(tokens, replacements)
            if len(re.findall(r"[A-Za-z]+", text)) < 3:
                return whole
            clauses.append((text, {t.i for t in tokens}))
        return clauses
