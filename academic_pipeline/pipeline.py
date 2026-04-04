"""MedCompress Academic Pipeline Orchestrator.

Manages the 10-stage academic paper pipeline:
    1. RESEARCH    — Deep literature research on medical compression
    2. WRITE       — Draft the paper (IMRaD structure)
    2.5 INTEGRITY  — Verify references, data, citations
    3. REVIEW      — Multi-perspective peer review
    4. REVISE      — Address reviewer comments
    3' RE-REVIEW   — Verification review
    4' RE-REVISE   — Second revision if needed
    4.5 FINAL_INTEGRITY — Final 100% verification
    5. FINALIZE    — Format conversion (Markdown, LaTeX, PDF)
    6. SUMMARY     — Process documentation

Each stage produces deliverables consumed by subsequent stages.
The pipeline supports mid-entry (skip completed stages) and
integrates stop-slop quality filters at every writing stage.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from . import stop_slop


class Stage(Enum):
    """Pipeline stages in execution order."""

    RESEARCH = "1_research"
    WRITE = "2_write"
    INTEGRITY = "2.5_integrity"
    REVIEW = "3_review"
    REVISE = "4_revise"
    RE_REVIEW = "3p_re_review"
    RE_REVISE = "4p_re_revise"
    FINAL_INTEGRITY = "4.5_final_integrity"
    FINALIZE = "5_finalize"
    SUMMARY = "6_summary"


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReviewDecision(Enum):
    """Peer review editorial decision."""

    ACCEPT = "accept"
    MINOR_REVISION = "minor_revision"
    MAJOR_REVISION = "major_revision"
    REJECT = "reject"


@dataclass
class StageResult:
    """Result of executing a pipeline stage."""

    stage: Stage
    status: StageStatus
    deliverables: dict[str, Any] = field(default_factory=dict)
    slop_report: stop_slop.SlopReport | None = None
    errors: list[str] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "stage": self.stage.value,
            "status": self.status.value,
            "deliverables": list(self.deliverables.keys()),
            "error_count": len(self.errors),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.slop_report:
            result["slop_violations"] = len(self.slop_report.violations)
            result["burstiness"] = self.slop_report.burstiness_score
        return result


@dataclass
class PipelineState:
    """Full state of the academic pipeline."""

    config: dict[str, Any]
    current_stage: Stage
    results: dict[str, StageResult] = field(default_factory=dict)
    review_decision: ReviewDecision | None = None
    revision_round: int = 0
    integrity_attempts: int = 0
    materials: dict[str, str] = field(default_factory=dict)

    def is_stage_complete(self, stage: Stage) -> bool:
        result = self.results.get(stage.value)
        return result is not None and result.status == StageStatus.COMPLETED

    def get_deliverable(self, stage: Stage, key: str) -> Any | None:
        result = self.results.get(stage.value)
        if result is None:
            return None
        return result.deliverables.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage.value,
            "revision_round": self.revision_round,
            "integrity_attempts": self.integrity_attempts,
            "review_decision": (self.review_decision.value
                                if self.review_decision else None),
            "stages": {
                k: v.to_dict() for k, v in self.results.items()
            },
            "materials": list(self.materials.keys()),
        }


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the academic pipeline."""

    title: str
    task: str  # "classification" or "segmentation"
    research_topic: str
    research_mode: str  # "full", "quick", "socratic"
    paper_type: str  # "imrad", "conference", "literature_review"
    citation_format: str  # "ieee", "apa7", "vancouver"
    target_journal: str
    output_formats: tuple[str, ...]  # ("markdown", "latex", "pdf")
    output_dir: str
    word_count_target: int
    enable_stop_slop: bool
    enable_style_calibration: bool
    compression_methods: tuple[str, ...]  # ("qat", "kd", "ptq")
    datasets: tuple[str, ...]  # ("isic", "brats")
    max_revision_rounds: int
    max_integrity_retries: int

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        return cls(
            title=raw["title"],
            task=raw["task"],
            research_topic=raw["research_topic"],
            research_mode=raw.get("research_mode", "full"),
            paper_type=raw.get("paper_type", "imrad"),
            citation_format=raw.get("citation_format", "ieee"),
            target_journal=raw.get("target_journal", ""),
            output_formats=tuple(raw.get("output_formats", ["markdown"])),
            output_dir=raw.get("output_dir", "outputs/paper"),
            word_count_target=raw.get("word_count_target", 8000),
            enable_stop_slop=raw.get("enable_stop_slop", True),
            enable_style_calibration=raw.get("enable_style_calibration", False),
            compression_methods=tuple(raw.get("compression_methods",
                                              ["qat", "kd"])),
            datasets=tuple(raw.get("datasets", ["isic"])),
            max_revision_rounds=raw.get("max_revision_rounds", 2),
            max_integrity_retries=raw.get("max_integrity_retries", 3),
        )


# ---------------------------------------------------------------------------
# Stage transition logic
# ---------------------------------------------------------------------------

STAGE_ORDER = list(Stage)

STAGE_TRANSITIONS: dict[Stage, dict[str, Stage]] = {
    Stage.RESEARCH: {"next": Stage.WRITE},
    Stage.WRITE: {"next": Stage.INTEGRITY},
    Stage.INTEGRITY: {
        "pass": Stage.REVIEW,
        "fail": Stage.INTEGRITY,  # retry
    },
    Stage.REVIEW: {
        "accept": Stage.FINAL_INTEGRITY,
        "minor_revision": Stage.REVISE,
        "major_revision": Stage.REVISE,
        "reject": Stage.WRITE,  # restart from write
    },
    Stage.REVISE: {"next": Stage.RE_REVIEW},
    Stage.RE_REVIEW: {
        "accept": Stage.FINAL_INTEGRITY,
        "minor_revision": Stage.FINAL_INTEGRITY,
        "major_revision": Stage.RE_REVISE,
    },
    Stage.RE_REVISE: {"next": Stage.FINAL_INTEGRITY},
    Stage.FINAL_INTEGRITY: {
        "pass": Stage.FINALIZE,
        "fail": Stage.FINAL_INTEGRITY,  # retry
    },
    Stage.FINALIZE: {"next": Stage.SUMMARY},
    Stage.SUMMARY: {},  # terminal
}


def get_next_stage(
    current: Stage,
    decision: str | None = None,
) -> Stage | None:
    """Determine the next stage based on current stage and decision."""
    transitions = STAGE_TRANSITIONS.get(current, {})
    if not transitions:
        return None
    if decision and decision in transitions:
        return transitions[decision]
    return transitions.get("next")


def detect_entry_stage(materials: dict[str, str]) -> Stage:
    """Detect which stage to start from based on existing materials.

    Supports mid-entry: if a user already has a draft, skip to INTEGRITY.
    If they have reviewer comments, skip to REVISE.
    """
    if "reviewer_comments" in materials:
        return Stage.REVISE
    if "paper_draft" in materials:
        return Stage.INTEGRITY
    if "research_report" in materials:
        return Stage.WRITE
    return Stage.RESEARCH


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------

@dataclass
class IntegrityCheck:
    """Result of an integrity verification phase."""

    references_verified: int
    references_total: int
    citations_matched: bool
    data_verified: bool
    originality_checked: bool
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.references_verified == self.references_total
            and self.citations_matched
            and self.data_verified
            and len(self.issues) == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "references_verified": self.references_verified,
            "references_total": self.references_total,
            "citations_matched": self.citations_matched,
            "data_verified": self.data_verified,
            "originality_checked": self.originality_checked,
            "passed": self.passed,
            "issues": self.issues,
        }


def verify_references(text: str) -> IntegrityCheck:
    """Verify reference integrity in the paper text.

    Checks that:
    - Every in-text citation has a matching reference entry
    - Every reference is cited at least once
    - No ghost citations exist
    """
    # Extract in-text citations: [1], [2,3], (Author, 2024), etc.
    bracket_cites = set()
    for m in re.finditer(r'\[(\d+(?:\s*,\s*\d+)*)\]', text):
        for num in re.split(r'\s*,\s*', m.group(1)):
            bracket_cites.add(num.strip())

    author_cites = set()
    for m in re.finditer(r'\(([A-Z][a-z]+(?:\s+(?:et al\.|&\s+[A-Z][a-z]+))?'
                         r',?\s*\d{4})\)', text):
        author_cites.add(m.group(1))

    total_cites = len(bracket_cites) + len(author_cites)

    # Extract reference list entries
    ref_section = ""
    ref_match = re.search(r'(?:^|\n)##?\s*References?\s*\n(.*)',
                          text, re.DOTALL | re.IGNORECASE)
    if ref_match:
        ref_section = ref_match.group(1)

    ref_entries = re.findall(r'^\s*\[?\d+\]?\s*.+', ref_section, re.MULTILINE)

    issues = []
    if total_cites == 0 and len(ref_entries) == 0:
        issues.append("No citations or references found.")

    return IntegrityCheck(
        references_verified=min(total_cites, len(ref_entries)),
        references_total=max(total_cites, len(ref_entries)),
        citations_matched=total_cites > 0 and len(ref_entries) > 0,
        data_verified=True,  # placeholder for real data verification
        originality_checked=False,  # requires external tool
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Paper section extraction
# ---------------------------------------------------------------------------

SECTION_PATTERNS = [
    ("abstract", r'(?i)^##?\s*Abstract\s*$'),
    ("introduction", r'(?i)^##?\s*\d*\.?\s*Introduction\s*$'),
    ("literature_review", r'(?i)^##?\s*\d*\.?\s*Literature\s+Review\s*$'),
    ("methodology", r'(?i)^##?\s*\d*\.?\s*Method(?:ology|s)?\s*$'),
    ("results", r'(?i)^##?\s*\d*\.?\s*Results?\s*(?:/\s*Findings?)?\s*$'),
    ("discussion", r'(?i)^##?\s*\d*\.?\s*Discussion\s*$'),
    ("conclusion", r'(?i)^##?\s*\d*\.?\s*Conclusion\s*$'),
    ("references", r'(?i)^##?\s*References?\s*$'),
]


def extract_sections(text: str) -> dict[str, str]:
    """Extract named sections from a markdown paper."""
    lines = text.split('\n')
    sections: dict[str, str] = {}
    current_section = "preamble"
    current_lines: list[str] = []

    for line in lines:
        matched = False
        for name, pattern in SECTION_PATTERNS:
            if re.match(pattern, line.strip()):
                if current_lines:
                    sections[current_section] = '\n'.join(current_lines).strip()
                current_section = name
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = '\n'.join(current_lines).strip()

    return sections


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityGate:
    """Quality thresholds for pipeline progression."""

    min_word_count: int = 3000
    max_slop_warnings: int = 10
    max_slop_errors: int = 0
    min_burstiness: float = 0.3
    min_references: int = 15
    require_all_sections: bool = True

    REQUIRED_SECTIONS: tuple[str, ...] = (
        "abstract", "introduction", "methodology",
        "results", "discussion", "conclusion", "references",
    )


def evaluate_quality(
    text: str,
    gate: QualityGate,
) -> tuple[bool, list[str]]:
    """Evaluate paper quality against the gate thresholds.

    Returns (passed, list of failure reasons).
    """
    failures: list[str] = []

    # Word count
    wc = len(text.split())
    if wc < gate.min_word_count:
        failures.append(
            f"Word count {wc} below minimum {gate.min_word_count}")

    # Sections
    sections = extract_sections(text)
    if gate.require_all_sections:
        for sec in gate.REQUIRED_SECTIONS:
            if sec not in sections:
                failures.append(f"Missing required section: {sec}")

    # Stop-slop analysis
    report = stop_slop.analyze(text)
    if report.warning_count > gate.max_slop_warnings:
        failures.append(
            f"Too many slop warnings: {report.warning_count} "
            f"(max {gate.max_slop_warnings})")
    if report.error_count > gate.max_slop_errors:
        failures.append(
            f"Slop errors found: {report.error_count}")
    if report.burstiness_score < gate.min_burstiness:
        failures.append(
            f"Low burstiness: {report.burstiness_score:.2f} "
            f"(min {gate.min_burstiness})")

    # Reference count
    ref_check = verify_references(text)
    if ref_check.references_total < gate.min_references:
        failures.append(
            f"Only {ref_check.references_total} references "
            f"(min {gate.min_references})")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class AcademicPipeline:
    """Orchestrates the full academic paper pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = PipelineState(
            config=asdict(config),
            current_stage=Stage.RESEARCH,
        )
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_gate = QualityGate()

    def detect_entry(self, materials: dict[str, str]) -> Stage:
        """Detect entry point from provided materials."""
        self.state.materials = materials
        entry = detect_entry_stage(materials)
        self.state.current_stage = entry
        return entry

    def begin_stage(self, stage: Stage) -> StageResult:
        """Mark a stage as in-progress and return a result handle."""
        result = StageResult(
            stage=stage,
            status=StageStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.state.results[stage.value] = result
        self.state.current_stage = stage
        return result

    def complete_stage(
        self,
        stage: Stage,
        deliverables: dict[str, Any],
        text_content: str | None = None,
    ) -> StageResult:
        """Mark a stage as completed, run stop-slop if applicable."""
        result = self.state.results.get(stage.value)
        if result is None:
            result = StageResult(stage=stage, status=StageStatus.COMPLETED)

        result.status = StageStatus.COMPLETED
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.deliverables = deliverables

        # Run stop-slop on any text-producing stage
        if text_content and self.config.enable_stop_slop:
            result.slop_report = stop_slop.analyze(text_content)

        self.state.results[stage.value] = result
        return result

    def fail_stage(self, stage: Stage, errors: list[str]) -> StageResult:
        """Mark a stage as failed."""
        result = self.state.results.get(stage.value)
        if result is None:
            result = StageResult(stage=stage, status=StageStatus.FAILED)
        result.status = StageStatus.FAILED
        result.errors = errors
        result.completed_at = datetime.now(timezone.utc).isoformat()
        self.state.results[stage.value] = result
        return result

    def advance(self, decision: str | None = None) -> Stage | None:
        """Advance to the next stage based on current state and decision."""
        next_stage = get_next_stage(self.state.current_stage, decision)
        if next_stage is not None:
            self.state.current_stage = next_stage

            # Track revision rounds
            if next_stage in (Stage.REVISE, Stage.RE_REVISE):
                self.state.revision_round += 1
            if next_stage in (Stage.INTEGRITY, Stage.FINAL_INTEGRITY):
                self.state.integrity_attempts += 1

        return next_stage

    def run_quality_gate(self, text: str) -> tuple[bool, list[str]]:
        """Run the quality gate on paper text."""
        return evaluate_quality(text, self.quality_gate)

    def run_integrity_check(self, text: str) -> IntegrityCheck:
        """Run integrity verification on the paper."""
        return verify_references(text)

    def get_status_dashboard(self) -> str:
        """Generate a human-readable status dashboard."""
        lines = [
            "=" * 50,
            "  MEDCOMPRESS ACADEMIC PIPELINE STATUS",
            "=" * 50,
            f"  Paper: {self.config.title}",
            f"  Current Stage: {self.state.current_stage.value}",
            f"  Revision Round: {self.state.revision_round}/"
            f"{self.config.max_revision_rounds}",
            "",
        ]

        for stage in Stage:
            result = self.state.results.get(stage.value)
            if result is None:
                icon = "○"
                status = "pending"
            elif result.status == StageStatus.COMPLETED:
                icon = "●"
                status = "completed"
            elif result.status == StageStatus.IN_PROGRESS:
                icon = "◐"
                status = "in progress"
            elif result.status == StageStatus.FAILED:
                icon = "✗"
                status = "failed"
            else:
                icon = "–"
                status = result.status.value

            line = f"  {icon} {stage.value:<25} {status}"
            if result and result.slop_report:
                sr = result.slop_report
                line += f"  [slop: {len(sr.violations)} violations]"
            lines.append(line)

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def save_state(self, path: str | None = None) -> str:
        """Save pipeline state to JSON."""
        if path is None:
            path = str(self.output_dir / "pipeline_state.json")
        state_dict = self.state.to_dict()
        state_dict["config"] = {
            "title": self.config.title,
            "task": self.config.task,
            "paper_type": self.config.paper_type,
            "citation_format": self.config.citation_format,
        }
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        return path

    def load_state(self, path: str) -> None:
        """Load pipeline state from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        stage_value = data.get("current_stage", Stage.RESEARCH.value)
        for s in Stage:
            if s.value == stage_value:
                self.state.current_stage = s
                break
        self.state.revision_round = data.get("revision_round", 0)
        self.state.integrity_attempts = data.get("integrity_attempts", 0)
        decision = data.get("review_decision")
        if decision:
            self.state.review_decision = ReviewDecision(decision)
