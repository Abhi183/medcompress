"""Stop-slop filter for human-like academic writing.

Detects and flags AI-typical writing patterns in academic prose.
Based on the Writing Quality Check protocol. This module does NOT
attempt to evade AI detectors — it enforces good writing rules
that produce clear, precise, varied academic prose.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """Violation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Violation:
    """A single writing quality violation."""

    rule: str
    severity: Severity
    line: int
    column: int
    message: str
    suggestion: str
    context: str  # surrounding text snippet


@dataclass(frozen=True)
class SlopReport:
    """Aggregated report of all violations found in a text."""

    violations: tuple[Violation, ...]
    word_count: int
    sentence_count: int
    paragraph_count: int
    em_dash_count: int
    burstiness_score: float  # 0.0 = monotonous, 1.0 = highly varied

    @property
    def is_clean(self) -> bool:
        return len(self.violations) == 0

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.INFO)


# ---------------------------------------------------------------------------
# A. High-Frequency Term Warnings
# ---------------------------------------------------------------------------

FLAGGED_TERMS: dict[str, dict[str, str]] = {
    "delve": {
        "reason": "Overused as 'explore' substitute",
        "alternatives": "examine, investigate, analyze, explore",
    },
    "tapestry": {
        "reason": "Cliché metaphor for complexity",
        "alternatives": "network, interplay, system",
    },
    "landscape": {
        "reason": "Vague when not literal",
        "alternatives": "field, domain, context, state of",
    },
    "pivotal": {
        "reason": "Inflation of importance",
        "alternatives": "important, significant, central, key",
    },
    "crucial": {
        "reason": "Same inflation pattern",
        "alternatives": "essential, necessary, critical, vital",
    },
    "foster": {
        "reason": "Vague verb",
        "alternatives": "promote, develop, cultivate, encourage",
    },
    "showcase": {
        "reason": "Non-academic register",
        "alternatives": "demonstrate, illustrate, present, reveal",
    },
    "testament": {
        "reason": "Cliché",
        "alternatives": "evidence, indicator, demonstration",
    },
    "navigate": {
        "reason": "Vague when not literal",
        "alternatives": "manage, address, handle, negotiate",
    },
    "leverage": {
        "reason": "Business jargon",
        "alternatives": "use, employ, utilize, apply",
    },
    "realm": {
        "reason": "Archaic/poetic",
        "alternatives": "domain, field, area, sphere",
    },
    "embark": {
        "reason": "Overwrought for 'begin'",
        "alternatives": "begin, initiate, undertake, start",
    },
    "underscore": {
        "reason": "Overused emphasis verb",
        "alternatives": "emphasize, highlight, stress, reinforce",
    },
    "multifaceted": {
        "reason": "Vague complexity claim",
        "alternatives": "complex, varied, diverse, multilayered",
    },
    "nuanced": {
        "reason": "Often vacuous",
        "alternatives": "subtle, detailed, fine-grained, qualified",
    },
    "comprehensive": {
        "reason": "Often unjustified",
        "alternatives": "thorough, extensive, broad, detailed",
    },
    "robust": {
        "reason": "Vague quality claim (exception: statistics)",
        "alternatives": "reliable, strong, rigorous, resilient",
    },
    "intricate": {
        "reason": "Same problem as multifaceted",
        "alternatives": "complex, detailed, elaborate, involved",
    },
    "cornerstone": {
        "reason": "Cliché metaphor",
        "alternatives": "foundation, basis, core element, pillar",
    },
    "paradigm": {
        "reason": "Overused outside philosophy of science",
        "alternatives": "framework, model, approach",
    },
    "synergy": {
        "reason": "Business jargon",
        "alternatives": "interaction, cooperation, combined effect",
    },
    "holistic": {
        "reason": "Vague without definition",
        "alternatives": "integrated, whole-system",
    },
    "streamline": {
        "reason": "Non-academic",
        "alternatives": "simplify, optimize, improve efficiency",
    },
    "cutting-edge": {
        "reason": "Cliché",
        "alternatives": "recent, advanced, state-of-the-art, novel",
    },
    "groundbreaking": {
        "reason": "Inflation",
        "alternatives": "novel, innovative, pioneering, original",
    },
}

# Technical terms that are exempt when used in their discipline context
TERM_EXEMPTIONS: dict[str, list[str]] = {
    "paradigm": ["paradigm shift", "kuhn", "philosophy of science"],
    "landscape": ["ecology", "geography", "geomorphology", "landscape ecology"],
    "robust": ["robust estimator", "robust regression", "robust statistics",
               "robust optimization"],
    "navigate": ["wayfinding", "navigation system", "spatial navigation"],
}

# ---------------------------------------------------------------------------
# B. Punctuation Pattern Control
# ---------------------------------------------------------------------------

EM_DASH_LIMIT = 3  # per paper total
SEMICOLON_LIMIT_PER_1K = 2  # per 1000 words

# ---------------------------------------------------------------------------
# C. Throat-Clearing Openers
# ---------------------------------------------------------------------------

THROAT_CLEARING_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)\bIn the realm of\b", "Delete. Start with the actual subject."),
    (r"(?i)\bIt(?:'s| is) important to note that\b",
     "Delete. The content speaks for itself."),
    (r"(?i)\bIt is worth mentioning that\b", "Delete. Just state it."),
    (r"(?i)\bIn today's rapidly evolving\b",
     "Delete. Timestamped clichés add no information."),
    (r"(?i)\bThis serves as a testament to\b",
     "Replace with 'This demonstrates...' or state the evidence directly."),
    (r"(?i)\bIt goes without saying that\b",
     "If it goes without saying, don't say it."),
    (r"(?i)\bIn order to\b", "Replace with 'To...'."),
    (r"(?i)\bIt should be noted that\b", "Delete. Just note it."),
    (r"(?i)\bAs a matter of fact\b", "Delete. State the fact."),
    (r"(?i)\bWhen it comes to\b",
     "Replace with the subject directly: 'X shows...'."),
    (r"(?i)\bAt the end of the day\b", "Delete. Colloquial and vague."),
    (r"(?i)\bWith that being said\b",
     "Delete or use 'However' if a contrast is intended."),
]

META_COMMENTARY_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)\bThis section will discuss\b",
     "Just discuss it. Remove meta-commentary."),
    (r"(?i)\bThe following paragraph examines\b",
     "Just examine it. Remove meta-commentary."),
    (r"(?i)\bWe now turn our attention to\b",
     "Just turn to it. Remove meta-commentary."),
    (r"(?i)\bThis paper aims to\b",
     "State the aim directly. 'We investigate...' or 'This study examines...'."),
    (r"(?i)\bThe purpose of this section is to\b",
     "Delete. Let the section speak for itself."),
]

# ---------------------------------------------------------------------------
# D. Structure Pattern Warnings
# ---------------------------------------------------------------------------

BINARY_CONTRAST_PATTERN = re.compile(
    r"(?i)(?:not\s+\w+[\.\,]\s+(?:but\s+)?(?:rather|instead))|"
    r"(?:it(?:'s| is) not about .{3,40}(?:—|--)\s*it(?:'s| is) about)"
)
BINARY_CONTRAST_LIMIT = 2  # per paper


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple heuristic."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines."""
    paras = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paras if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _compute_burstiness(sentence_lengths: list[int]) -> float:
    """Compute burstiness score from sentence word counts.

    Returns a value between 0.0 (perfectly monotonous) and 1.0 (highly varied).
    Uses coefficient of variation normalized to [0, 1].
    """
    if len(sentence_lengths) < 3:
        return 1.0  # too few sentences to judge
    mean = sum(sentence_lengths) / len(sentence_lengths)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in sentence_lengths) / len(sentence_lengths)
    std_dev = variance ** 0.5
    cv = std_dev / mean
    # Normalize: CV of 0.3+ is considered good variation
    return min(cv / 0.3, 1.0)


def _find_line_col(text: str, match_start: int) -> tuple[int, int]:
    """Convert character offset to (line, column) 1-indexed."""
    prefix = text[:match_start]
    line = prefix.count('\n') + 1
    last_nl = prefix.rfind('\n')
    col = match_start - last_nl if last_nl >= 0 else match_start + 1
    return line, col


def _get_context(text: str, start: int, end: int, window: int = 40) -> str:
    """Extract a snippet of surrounding text for context."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    snippet = text[ctx_start:ctx_end].replace('\n', ' ')
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _is_exempt(term: str, text: str) -> bool:
    """Check if a flagged term is used in an exempt discipline context."""
    exemptions = TERM_EXEMPTIONS.get(term, [])
    text_lower = text.lower()
    return any(ex in text_lower for ex in exemptions)


def check_flagged_terms(text: str) -> list[Violation]:
    """Check for overused AI-typical terms."""
    violations = []
    text_lower = text.lower()
    for term, info in FLAGGED_TERMS.items():
        if _is_exempt(term, text):
            continue
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            line, col = _find_line_col(text, match.start())
            violations.append(Violation(
                rule="flagged_term",
                severity=Severity.WARNING,
                line=line,
                column=col,
                message=f"Flagged term '{term}': {info['reason']}",
                suggestion=f"Consider: {info['alternatives']}",
                context=_get_context(text, match.start(), match.end()),
            ))
    return violations


def check_punctuation(text: str) -> list[Violation]:
    """Check em-dash and semicolon overuse."""
    violations = []
    wc = _word_count(text)

    # Em dashes (—, --, or –)
    em_dashes = list(re.finditer(r'—|--|–', text))
    if len(em_dashes) > EM_DASH_LIMIT:
        for idx, match in enumerate(em_dashes[EM_DASH_LIMIT:],
                                    start=EM_DASH_LIMIT + 1):
            line, col = _find_line_col(text, match.start())
            violations.append(Violation(
                rule="em_dash_overuse",
                severity=Severity.WARNING,
                line=line,
                column=col,
                message=(f"Em dash #{idx} of "
                         f"{len(em_dashes)} (limit: {EM_DASH_LIMIT})"),
                suggestion="Replace with commas, parentheses, or restructure.",
                context=_get_context(text, match.start(), match.end()),
            ))

    # Semicolons
    semicolons = list(re.finditer(r';', text))
    limit = max(1, int(wc / 1000 * SEMICOLON_LIMIT_PER_1K))
    if len(semicolons) > limit:
        for idx, match in enumerate(semicolons[limit:], start=limit + 1):
            line, col = _find_line_col(text, match.start())
            violations.append(Violation(
                rule="semicolon_overuse",
                severity=Severity.INFO,
                line=line,
                column=col,
                message=(f"Semicolon #{idx} of "
                         f"{len(semicolons)} (limit: ~{limit} for {wc} words)"),
                suggestion="Use a period and start a new sentence.",
                context=_get_context(text, match.start(), match.end()),
            ))

    return violations


def check_throat_clearing(text: str) -> list[Violation]:
    """Check for throat-clearing openers and meta-commentary."""
    violations = []
    all_patterns = THROAT_CLEARING_PATTERNS + META_COMMENTARY_PATTERNS
    for pattern_str, fix in all_patterns:
        pattern = re.compile(pattern_str)
        for match in pattern.finditer(text):
            line, col = _find_line_col(text, match.start())
            violations.append(Violation(
                rule="throat_clearing",
                severity=Severity.WARNING,
                line=line,
                column=col,
                message=f"Throat-clearing opener: '{match.group()}'",
                suggestion=fix,
                context=_get_context(text, match.start(), match.end()),
            ))
    return violations


def check_structure_patterns(text: str) -> list[Violation]:
    """Check for structural monotony and binary contrast overuse."""
    violations = []

    # Binary contrast overuse
    contrasts = list(BINARY_CONTRAST_PATTERN.finditer(text))
    if len(contrasts) > BINARY_CONTRAST_LIMIT:
        for idx, match in enumerate(contrasts[BINARY_CONTRAST_LIMIT:],
                                    start=BINARY_CONTRAST_LIMIT + 1):
            line, col = _find_line_col(text, match.start())
            violations.append(Violation(
                rule="binary_contrast_overuse",
                severity=Severity.INFO,
                line=line,
                column=col,
                message=(f"Binary contrast #{idx} of "
                         f"{len(contrasts)} (limit: {BINARY_CONTRAST_LIMIT})"),
                suggestion="This rhetorical device loses impact with repetition.",
                context=_get_context(text, match.start(), match.end()),
            ))

    # Uniform paragraph length
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) >= 4:
        para_lengths = [_word_count(p) for p in paragraphs]
        mean_len = sum(para_lengths) / len(para_lengths)
        if mean_len > 0:
            deviations = [abs(plen - mean_len) / mean_len for plen in para_lengths]
            avg_deviation = sum(deviations) / len(deviations)
            if avg_deviation < 0.15:  # all paragraphs within 15% of mean
                violations.append(Violation(
                    rule="uniform_paragraph_length",
                    severity=Severity.INFO,
                    line=1,
                    column=1,
                    message=("Paragraphs have uniform length "
                             f"(avg deviation: {avg_deviation:.0%}). "
                             "Natural writing varies paragraph length."),
                    suggestion=("Mix short paragraphs (2-3 sentences) with "
                                "longer ones (6-8 sentences) for rhythm."),
                    context=f"Paragraph word counts: {para_lengths[:8]}",
                ))

    return violations


def check_burstiness(text: str) -> list[Violation]:
    """Check sentence length variation (burstiness)."""
    violations = []
    sentences = _split_sentences(text)
    if len(sentences) < 5:
        return violations

    lengths = [_word_count(s) for s in sentences]

    # Check for 5+ consecutive sentences in narrow range
    window = 5
    for i in range(len(lengths) - window + 1):
        window_lengths = lengths[i:i + window]
        min_len = min(window_lengths)
        max_len = max(window_lengths)
        if max_len > 0 and (max_len - min_len) / max_len < 0.25:
            # Find line of the first sentence in the window
            offset = sum(len(s) + 1 for s in sentences[:i])
            line, col = _find_line_col(text, min(offset, len(text) - 1))
            violations.append(Violation(
                rule="low_burstiness",
                severity=Severity.WARNING,
                line=line,
                column=col,
                message=(f"5 consecutive sentences with similar length "
                         f"({min_len}-{max_len} words). "
                         "This creates a monotonous rhythm."),
                suggestion=("Insert a short sentence (≤10 words) or "
                            "combine two sentences for variation."),
                context=f"Sentence lengths: {window_lengths}",
            ))
            break  # report once per text block

    return violations


def check_synonym_cycling(text: str) -> list[Violation]:
    """Detect synonym cycling within paragraphs.

    Looks for the pattern where 3+ different terms for the same concept
    appear within a single paragraph. Uses a curated set of common
    synonym groups in medical imaging compression.
    """
    violations = []
    synonym_groups: list[tuple[str, ...]] = [
        ("model", "network", "architecture", "system"),
        ("compression", "pruning", "quantization", "distillation"),
        ("accuracy", "performance", "efficacy", "effectiveness"),
        ("method", "approach", "technique", "strategy"),
        ("patients", "subjects", "participants", "individuals"),
        ("images", "scans", "acquisitions", "data"),
        ("reduce", "decrease", "diminish", "lower", "minimize"),
        ("improve", "enhance", "augment", "boost", "elevate"),
    ]

    paragraphs = _split_paragraphs(text)
    for para_idx, para in enumerate(paragraphs):
        para_lower = para.lower()
        for group in synonym_groups:
            found = [t for t in group if re.search(r'\b' + t + r'\b', para_lower)]
            if len(found) >= 3:
                offset = text.find(para)
                line, col = _find_line_col(text, max(0, offset))
                violations.append(Violation(
                    rule="synonym_cycling",
                    severity=Severity.INFO,
                    line=line,
                    column=col,
                    message=(f"Synonym cycling in paragraph {para_idx + 1}: "
                             f"{', '.join(found)}. "
                             "Pick one term and use it consistently."),
                    suggestion=("In academic writing, consistent terminology "
                                "is a virtue. Repeat the best term."),
                    context=para[:120] + "...",
                ))
    return violations


def analyze(text: str, *, section_name: str | None = None) -> SlopReport:
    """Run all stop-slop checks on the given text.

    Args:
        text: The academic text to analyze.
        section_name: Optional section identifier. Methods sections
            tolerate lower burstiness; abstracts skip structure checks.

    Returns:
        A SlopReport with all violations and text metrics.
    """
    violations: list[Violation] = []
    violations.extend(check_flagged_terms(text))
    violations.extend(check_punctuation(text))
    violations.extend(check_throat_clearing(text))

    # Methods sections have naturally uniform structure — relax checks
    if section_name not in ("methodology", "methods"):
        violations.extend(check_structure_patterns(text))

    # Burstiness check on all sections
    violations.extend(check_burstiness(text))
    violations.extend(check_synonym_cycling(text))

    # Sort by line number, then column
    violations.sort(key=lambda v: (v.line, v.column))

    sentences = _split_sentences(text)
    sentence_lengths = [_word_count(s) for s in sentences]
    burstiness = _compute_burstiness(sentence_lengths)

    em_dashes = len(re.findall(r'—|--|–', text))

    return SlopReport(
        violations=tuple(violations),
        word_count=_word_count(text),
        sentence_count=len(sentences),
        paragraph_count=len(_split_paragraphs(text)),
        em_dash_count=em_dashes,
        burstiness_score=burstiness,
    )


def format_report(report: SlopReport) -> str:
    """Format a SlopReport as a human-readable string."""
    lines = [
        "=" * 60,
        "  STOP-SLOP WRITING QUALITY REPORT",
        "=" * 60,
        "",
        f"Words: {report.word_count}  |  "
        f"Sentences: {report.sentence_count}  |  "
        f"Paragraphs: {report.paragraph_count}",
        f"Em dashes: {report.em_dash_count}/{EM_DASH_LIMIT}  |  "
        f"Burstiness: {report.burstiness_score:.2f}/1.00",
        "",
    ]

    if report.is_clean:
        lines.append("✓ No violations found. Text passes quality check.")
    else:
        lines.append(
            f"Found {len(report.violations)} violation(s): "
            f"{report.error_count} error, "
            f"{report.warning_count} warning, "
            f"{report.info_count} info"
        )
        lines.append("")

        for v in report.violations:
            icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}[v.severity.value]
            lines.append(f"  {icon} L{v.line}:C{v.column} [{v.rule}]")
            lines.append(f"    {v.message}")
            lines.append(f"    → {v.suggestion}")
            lines.append(f"    Context: {v.context}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
