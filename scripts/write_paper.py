"""MedCompress Academic Paper Pipeline CLI.

Entry point for running the full academic paper pipeline.
Supports both full pipeline execution and individual stage runs.

Usage:
    python scripts/write_paper.py --config academic_pipeline/configs/medcompress_paper.yaml
    python scripts/write_paper.py --config academic_pipeline/configs/medcompress_paper.yaml --stage research
    python scripts/write_paper.py --analyze outputs/paper/draft.md
    python scripts/write_paper.py --status outputs/paper/pipeline_state.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from academic_pipeline import stop_slop
from academic_pipeline.pipeline import (
    AcademicPipeline,
    PipelineConfig,
    QualityGate,
    Stage,
    evaluate_quality,
    extract_sections,
    verify_references,
)


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline or a specific stage."""
    config = PipelineConfig.from_yaml(args.config)
    pipeline = AcademicPipeline(config)

    if args.resume:
        pipeline.load_state(args.resume)
        print(f"Resumed from: {args.resume}")
        print(f"Current stage: {pipeline.state.current_stage.value}")

    if args.stage:
        stage_map = {s.value.split("_", 1)[-1]: s for s in Stage}
        stage_map.update({s.name.lower(): s for s in Stage})
        target = stage_map.get(args.stage.lower())
        if target is None:
            print(f"Unknown stage: {args.stage}")
            print(f"Available: {', '.join(stage_map.keys())}")
            return 1
        pipeline.state.current_stage = target

    print(pipeline.get_status_dashboard())

    # Save initial state
    state_path = pipeline.save_state()
    print(f"\nPipeline state saved to: {state_path}")
    print("\nTo execute each stage, use the academic-pipeline skill")
    print("in Claude Code with the MedCompress configuration.")
    print("\nStage execution order:")
    for stage in Stage:
        print(f"  {stage.value}")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run stop-slop analysis on a paper draft."""
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    print(f"Analyzing: {path} ({len(text.split())} words)\n")

    # Full stop-slop analysis
    report = stop_slop.analyze(text)
    print(stop_slop.format_report(report))

    # Section-level analysis
    sections = extract_sections(text)
    if sections:
        print("\n\nPER-SECTION ANALYSIS:")
        print("-" * 50)
        for name, content in sections.items():
            if name in ("preamble", "references"):
                continue
            sec_report = stop_slop.analyze(content, section_name=name)
            status = "CLEAN" if sec_report.is_clean else f"{len(sec_report.violations)} issues"
            print(f"  {name:<20} {sec_report.word_count:>5} words  "
                  f"burstiness={sec_report.burstiness_score:.2f}  {status}")

    # Quality gate
    print("\n\nQUALITY GATE:")
    print("-" * 50)
    gate = QualityGate()
    passed, failures = evaluate_quality(text, gate)
    if passed:
        print("  PASSED - Paper meets all quality thresholds.")
    else:
        print("  FAILED:")
        for f in failures:
            print(f"    ✗ {f}")

    # Integrity check
    print("\n\nINTEGRITY CHECK:")
    print("-" * 50)
    integrity = verify_references(text)
    print(f"  References verified: {integrity.references_verified}/"
          f"{integrity.references_total}")
    print(f"  Citations matched: {integrity.citations_matched}")
    print(f"  Data verified: {integrity.data_verified}")
    if integrity.issues:
        for issue in integrity.issues:
            print(f"  ⚠ {issue}")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show pipeline status from saved state."""
    path = Path(args.file)
    if not path.exists():
        print(f"State file not found: {path}")
        return 1

    # Minimal config to create pipeline
    config = PipelineConfig(
        title="(loaded from state)",
        task="classification",
        research_topic="",
        research_mode="full",
        paper_type="imrad",
        citation_format="ieee",
        target_journal="",
        output_formats=("markdown",),
        output_dir=str(path.parent),
        word_count_target=8000,
        enable_stop_slop=True,
        enable_style_calibration=False,
        compression_methods=("qat", "kd"),
        datasets=("isic",),
        max_revision_rounds=2,
        max_integrity_retries=3,
    )
    pipeline = AcademicPipeline(config)
    pipeline.load_state(str(path))
    print(pipeline.get_status_dashboard())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MedCompress Academic Paper Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument("--config", required=True,
                            help="Path to pipeline YAML config")
    run_parser.add_argument("--stage", default=None,
                            help="Run a specific stage only")
    run_parser.add_argument("--resume", default=None,
                            help="Resume from saved state JSON")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run stop-slop analysis on a paper draft")
    analyze_parser.add_argument("file", help="Path to markdown paper file")

    # status command
    status_parser = subparsers.add_parser(
        "status", help="Show pipeline status")
    status_parser.add_argument("file", help="Path to pipeline_state.json")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
