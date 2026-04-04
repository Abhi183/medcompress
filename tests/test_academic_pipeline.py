"""Tests for the academic pipeline orchestrator.

Covers pipeline configuration, stage transitions, state management,
integrity verification, quality gates, section extraction, and the
full pipeline lifecycle.
"""

import json
import os
import tempfile

import pytest
import yaml

from academic_pipeline.pipeline import (
    AcademicPipeline,
    IntegrityCheck,
    PipelineConfig,
    PipelineState,
    QualityGate,
    ReviewDecision,
    Stage,
    StageResult,
    StageStatus,
    detect_entry_stage,
    evaluate_quality,
    extract_sections,
    get_next_stage,
    verify_references,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_config_dict():
    return {
        "title": "Test Paper on Compression",
        "task": "classification",
        "research_topic": "Medical imaging compression",
        "research_mode": "quick",
        "paper_type": "imrad",
        "citation_format": "ieee",
        "target_journal": "Test Journal",
        "output_formats": ["markdown"],
        "output_dir": tempfile.mkdtemp(),
        "word_count_target": 5000,
        "enable_stop_slop": True,
        "enable_style_calibration": False,
        "compression_methods": ["qat", "kd"],
        "datasets": ["isic"],
        "max_revision_rounds": 2,
        "max_integrity_retries": 3,
    }


@pytest.fixture
def sample_config(sample_config_dict):
    return PipelineConfig(
        title=sample_config_dict["title"],
        task=sample_config_dict["task"],
        research_topic=sample_config_dict["research_topic"],
        research_mode=sample_config_dict["research_mode"],
        paper_type=sample_config_dict["paper_type"],
        citation_format=sample_config_dict["citation_format"],
        target_journal=sample_config_dict["target_journal"],
        output_formats=tuple(sample_config_dict["output_formats"]),
        output_dir=sample_config_dict["output_dir"],
        word_count_target=sample_config_dict["word_count_target"],
        enable_stop_slop=sample_config_dict["enable_stop_slop"],
        enable_style_calibration=sample_config_dict["enable_style_calibration"],
        compression_methods=tuple(sample_config_dict["compression_methods"]),
        datasets=tuple(sample_config_dict["datasets"]),
        max_revision_rounds=sample_config_dict["max_revision_rounds"],
        max_integrity_retries=sample_config_dict["max_integrity_retries"],
    )


@pytest.fixture
def pipeline(sample_config):
    return AcademicPipeline(sample_config)


@pytest.fixture
def sample_paper_text():
    return """
## Abstract

Model compression enables deployment of medical imaging AI on mobile devices.
We evaluate quantization-aware training and knowledge distillation on two
clinical benchmarks: ISIC 2020 melanoma classification and BraTS 2021 brain
tumor segmentation. QAT achieves 4x compression with less than 2% AUC loss.
Knowledge distillation produces lightweight students retaining 95% of teacher
accuracy.

## 1. Introduction

Medical imaging AI models achieve strong diagnostic accuracy but require
significant computational resources [1]. Deploying these models on mobile
devices and web browsers enables point-of-care diagnostics in resource-limited
settings [2]. Model compression techniques reduce model size and inference
latency while preserving accuracy [3].

This paper presents MedCompress, an open-source benchmark for compressing
medical imaging models targeting mobile and WebAssembly endpoints. We evaluate
two compression methods: quantization-aware training (QAT) and knowledge
distillation (KD) on ISIC 2020 and BraTS 2021 datasets [4,5].

## 3. Methodology

We implement a compression pipeline with the following stages. First, we
train baseline models on each dataset. Then we apply QAT or KD compression.
Finally, we export to TFLite and ONNX formats for deployment [6,7,8].

### 3.1 Datasets

ISIC 2020 provides 33,126 dermoscopy images for binary melanoma classification
[9]. BraTS 2021 provides multi-modal brain MRI volumes for tumor segmentation
[10,11].

### 3.2 Baseline Models

For classification, we use EfficientNetB0 with transfer learning from ImageNet
[12]. For segmentation, we use a 4-stage U-Net with Dice-CrossEntropy loss [13].

## 4. Results

QAT INT8 achieves 4.1x compression ratio on ISIC classification with AUC
reduction from 0.92 to 0.90 [14]. KD produces a student model with 8x fewer
parameters achieving Dice coefficient of 0.82 versus 0.87 for the teacher on
BraTS segmentation [15,16].

## 5. Discussion

The results demonstrate that model compression is viable for medical imaging
applications. QAT works well for classification tasks where global features
matter. KD is more suitable for segmentation where spatial precision is
critical [17,18].

## 6. Conclusion

MedCompress provides an open-source benchmark for medical imaging compression.
Our evaluation shows that QAT and KD can significantly reduce model size
while preserving clinically useful accuracy for mobile deployment [19,20].

## References

[1] Author A, "Medical imaging AI," Journal, 2023.
[2] Author B, "Mobile health deployment," Conference, 2023.
[3] Author C, "Model compression survey," Review, 2024.
[4] Author D, "ISIC 2020 challenge," Dataset, 2020.
[5] Author E, "BraTS 2021 benchmark," Challenge, 2021.
[6] Author F, "TensorFlow Lite," Google, 2023.
[7] Author G, "ONNX format," Microsoft, 2023.
[8] Author H, "Quantization techniques," ICLR, 2022.
[9] Author I, "Dermoscopy analysis," Nature Med, 2020.
[10] Author J, "Brain tumor segmentation," NeuroImage, 2021.
[11] Author K, "BraTS evaluation," MICCAI, 2021.
[12] Author L, "EfficientNet," ICML, 2019.
[13] Author M, "U-Net architecture," MICCAI, 2015.
[14] Author N, "QAT for medical imaging," ISBI, 2024.
[15] Author O, "Knowledge distillation," NeurIPS, 2015.
[16] Author P, "Medical KD," TMI, 2024.
[17] Author Q, "Compression trade-offs," CVPR, 2023.
[18] Author R, "Segmentation compression," MICCAI, 2024.
[19] Author S, "Mobile medical AI," Lancet Digital, 2024.
[20] Author T, "Edge deployment review," ACM Computing, 2024.
"""


# -----------------------------------------------------------------------
# Config Tests
# -----------------------------------------------------------------------

class TestPipelineConfig:
    """Tests for pipeline configuration loading and validation."""

    def test_from_yaml(self, sample_config_dict):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(sample_config_dict, f)
            f.flush()
            config = PipelineConfig.from_yaml(f.name)

        assert config.title == "Test Paper on Compression"
        assert config.task == "classification"
        assert config.compression_methods == ("qat", "kd")
        assert config.enable_stop_slop is True
        os.unlink(f.name)

    def test_default_values(self):
        minimal = {
            "title": "Minimal",
            "task": "classification",
            "research_topic": "Testing",
        }
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(minimal, f)
            f.flush()
            config = PipelineConfig.from_yaml(f.name)

        assert config.research_mode == "full"
        assert config.paper_type == "imrad"
        assert config.citation_format == "ieee"
        assert config.enable_stop_slop is True
        os.unlink(f.name)

    def test_config_is_frozen(self, sample_config):
        with pytest.raises(AttributeError):
            sample_config.title = "Changed"


# -----------------------------------------------------------------------
# Stage Transition Tests
# -----------------------------------------------------------------------

class TestStageTransitions:
    """Tests for pipeline stage transition logic."""

    def test_research_to_write(self):
        assert get_next_stage(Stage.RESEARCH) == Stage.WRITE

    def test_write_to_integrity(self):
        assert get_next_stage(Stage.WRITE) == Stage.INTEGRITY

    def test_integrity_pass_to_review(self):
        assert get_next_stage(Stage.INTEGRITY, "pass") == Stage.REVIEW

    def test_integrity_fail_retries(self):
        assert get_next_stage(Stage.INTEGRITY, "fail") == Stage.INTEGRITY

    def test_review_accept_to_final_integrity(self):
        assert get_next_stage(Stage.REVIEW, "accept") == Stage.FINAL_INTEGRITY

    def test_review_minor_to_revise(self):
        assert get_next_stage(Stage.REVIEW, "minor_revision") == Stage.REVISE

    def test_review_major_to_revise(self):
        assert get_next_stage(Stage.REVIEW, "major_revision") == Stage.REVISE

    def test_review_reject_to_write(self):
        assert get_next_stage(Stage.REVIEW, "reject") == Stage.WRITE

    def test_revise_to_re_review(self):
        assert get_next_stage(Stage.REVISE) == Stage.RE_REVIEW

    def test_re_review_accept_to_final(self):
        assert get_next_stage(Stage.RE_REVIEW, "accept") == Stage.FINAL_INTEGRITY

    def test_re_review_major_to_re_revise(self):
        assert get_next_stage(Stage.RE_REVIEW, "major_revision") == Stage.RE_REVISE

    def test_re_revise_to_final_integrity(self):
        assert get_next_stage(Stage.RE_REVISE) == Stage.FINAL_INTEGRITY

    def test_final_integrity_pass_to_finalize(self):
        assert get_next_stage(Stage.FINAL_INTEGRITY, "pass") == Stage.FINALIZE

    def test_finalize_to_summary(self):
        assert get_next_stage(Stage.FINALIZE) == Stage.SUMMARY

    def test_summary_is_terminal(self):
        assert get_next_stage(Stage.SUMMARY) is None


# -----------------------------------------------------------------------
# Entry Point Detection Tests
# -----------------------------------------------------------------------

class TestEntryDetection:
    """Tests for mid-pipeline entry detection."""

    def test_no_materials_starts_at_research(self):
        assert detect_entry_stage({}) == Stage.RESEARCH

    def test_research_report_starts_at_write(self):
        assert detect_entry_stage({"research_report": "..."}) == Stage.WRITE

    def test_paper_draft_starts_at_integrity(self):
        assert detect_entry_stage({"paper_draft": "..."}) == Stage.INTEGRITY

    def test_reviewer_comments_starts_at_revise(self):
        assert detect_entry_stage({"reviewer_comments": "..."}) == Stage.REVISE

    def test_draft_and_comments_prefers_revise(self):
        materials = {"paper_draft": "...", "reviewer_comments": "..."}
        assert detect_entry_stage(materials) == Stage.REVISE


# -----------------------------------------------------------------------
# Integrity Verification Tests
# -----------------------------------------------------------------------

class TestIntegrityVerification:
    """Tests for reference and citation integrity checking."""

    def test_bracket_citations_detected(self):
        text = (
            "Method works well [1]. Results confirm [2,3].\n\n"
            "## References\n\n"
            "[1] Author A, Title, 2024.\n"
            "[2] Author B, Title, 2024.\n"
            "[3] Author C, Title, 2024.\n"
        )
        check = verify_references(text)
        assert check.references_verified >= 3
        assert check.citations_matched is True

    def test_author_citations_detected(self):
        text = (
            "According to Smith (2024), compression works. "
            "Also (Jones et al., 2023).\n\n"
            "## References\n\n"
            "[1] Smith, Title, 2024.\n"
            "[2] Jones et al., Title, 2023.\n"
        )
        check = verify_references(text)
        assert check.citations_matched is True

    def test_no_references_fails(self):
        text = "A paper with no citations or references at all."
        check = verify_references(text)
        assert check.passed is False
        assert len(check.issues) > 0

    def test_integrity_check_serializable(self):
        check = IntegrityCheck(
            references_verified=10,
            references_total=10,
            citations_matched=True,
            data_verified=True,
            originality_checked=False,
        )
        d = check.to_dict()
        assert d["passed"] is True
        assert d["references_verified"] == 10


# -----------------------------------------------------------------------
# Section Extraction Tests
# -----------------------------------------------------------------------

class TestSectionExtraction:
    """Tests for extracting named sections from markdown papers."""

    def test_extracts_standard_sections(self, sample_paper_text):
        sections = extract_sections(sample_paper_text)
        assert "abstract" in sections
        assert "introduction" in sections
        assert "methodology" in sections
        assert "results" in sections
        assert "discussion" in sections
        assert "conclusion" in sections
        assert "references" in sections

    def test_section_content_not_empty(self, sample_paper_text):
        sections = extract_sections(sample_paper_text)
        for name, content in sections.items():
            if name != "preamble":
                assert len(content.strip()) > 0, f"Section '{name}' is empty"

    def test_handles_missing_sections(self):
        text = "## Abstract\n\nSome abstract text.\n\n## References\n\n[1] Ref."
        sections = extract_sections(text)
        assert "abstract" in sections
        assert "references" in sections
        assert "methodology" not in sections


# -----------------------------------------------------------------------
# Quality Gate Tests
# -----------------------------------------------------------------------

class TestQualityGate:
    """Tests for the quality gate evaluation."""

    def test_good_paper_passes(self, sample_paper_text):
        gate = QualityGate(
            min_word_count=100,
            max_slop_warnings=50,
            min_burstiness=0.1,
            min_references=5,
        )
        passed, failures = evaluate_quality(sample_paper_text, gate)
        assert passed is True, f"Failures: {failures}"

    def test_short_paper_fails(self):
        text = "## Abstract\n\nToo short."
        gate = QualityGate(min_word_count=5000)
        passed, failures = evaluate_quality(text, gate)
        assert passed is False
        assert any("Word count" in f for f in failures)

    def test_missing_sections_fails(self):
        text = "## Abstract\n\nJust an abstract.\n" + " word" * 5000
        gate = QualityGate(
            min_word_count=100,
            require_all_sections=True,
        )
        passed, failures = evaluate_quality(text, gate)
        assert passed is False
        assert any("Missing required section" in f for f in failures)

    def test_low_references_fails(self):
        text = (
            "## Abstract\n\nAbstract text.\n\n"
            "## 1. Introduction\n\nIntro text [1].\n\n"
            "## 3. Methodology\n\nMethods.\n\n"
            "## 4. Results\n\nResults.\n\n"
            "## 5. Discussion\n\nDiscussion.\n\n"
            "## 6. Conclusion\n\nConclusion.\n\n"
            "## References\n\n[1] Single ref.\n"
            + " word" * 3000
        )
        gate = QualityGate(min_word_count=100, min_references=20)
        passed, failures = evaluate_quality(text, gate)
        assert passed is False
        assert any("reference" in f.lower() for f in failures)


# -----------------------------------------------------------------------
# Pipeline Lifecycle Tests
# -----------------------------------------------------------------------

class TestPipelineLifecycle:
    """Tests for the full pipeline orchestration."""

    def test_initial_state(self, pipeline):
        assert pipeline.state.current_stage == Stage.RESEARCH
        assert pipeline.state.revision_round == 0

    def test_begin_stage(self, pipeline):
        result = pipeline.begin_stage(Stage.RESEARCH)
        assert result.status == StageStatus.IN_PROGRESS
        assert result.started_at is not None

    def test_complete_stage(self, pipeline):
        pipeline.begin_stage(Stage.RESEARCH)
        result = pipeline.complete_stage(
            Stage.RESEARCH,
            deliverables={"bibliography": "refs.md"},
        )
        assert result.status == StageStatus.COMPLETED
        assert result.completed_at is not None
        assert "bibliography" in result.deliverables

    def test_complete_stage_with_slop_check(self, pipeline):
        pipeline.begin_stage(Stage.WRITE)
        text = "We delve into the pivotal compression landscape."
        result = pipeline.complete_stage(
            Stage.WRITE,
            deliverables={"draft": "draft.md"},
            text_content=text,
        )
        assert result.slop_report is not None
        assert result.slop_report.warning_count > 0

    def test_slop_check_disabled(self, sample_config_dict):
        sample_config_dict["enable_stop_slop"] = False
        config = PipelineConfig(
            title=sample_config_dict["title"],
            task="classification",
            research_topic="test",
            research_mode="quick",
            paper_type="imrad",
            citation_format="ieee",
            target_journal="",
            output_formats=("markdown",),
            output_dir=sample_config_dict["output_dir"],
            word_count_target=5000,
            enable_stop_slop=False,
            enable_style_calibration=False,
            compression_methods=("qat",),
            datasets=("isic",),
            max_revision_rounds=2,
            max_integrity_retries=3,
        )
        p = AcademicPipeline(config)
        p.begin_stage(Stage.WRITE)
        result = p.complete_stage(
            Stage.WRITE,
            deliverables={"draft": "draft.md"},
            text_content="We delve into the pivotal landscape.",
        )
        assert result.slop_report is None

    def test_fail_stage(self, pipeline):
        pipeline.begin_stage(Stage.INTEGRITY)
        result = pipeline.fail_stage(
            Stage.INTEGRITY,
            errors=["Ghost citation found"],
        )
        assert result.status == StageStatus.FAILED
        assert len(result.errors) == 1

    def test_advance_through_pipeline(self, pipeline):
        pipeline.state.current_stage = Stage.RESEARCH
        next_s = pipeline.advance()
        assert next_s == Stage.WRITE
        assert pipeline.state.current_stage == Stage.WRITE

        next_s = pipeline.advance()
        assert next_s == Stage.INTEGRITY

    def test_advance_with_review_decision(self, pipeline):
        pipeline.state.current_stage = Stage.REVIEW
        next_s = pipeline.advance("minor_revision")
        assert next_s == Stage.REVISE
        assert pipeline.state.revision_round == 1

    def test_revision_round_tracking(self, pipeline):
        pipeline.state.current_stage = Stage.REVIEW
        pipeline.advance("major_revision")
        assert pipeline.state.revision_round == 1

        pipeline.state.current_stage = Stage.RE_REVIEW
        pipeline.advance("major_revision")
        assert pipeline.state.revision_round == 2

    def test_integrity_attempt_tracking(self, pipeline):
        pipeline.state.current_stage = Stage.WRITE
        pipeline.advance()  # -> INTEGRITY
        assert pipeline.state.integrity_attempts == 1

    def test_status_dashboard(self, pipeline):
        dashboard = pipeline.get_status_dashboard()
        assert "MEDCOMPRESS" in dashboard
        assert "1_research" in dashboard

    def test_detect_entry_with_materials(self, pipeline):
        stage = pipeline.detect_entry({"paper_draft": "..."})
        assert stage == Stage.INTEGRITY
        assert pipeline.state.current_stage == Stage.INTEGRITY


# -----------------------------------------------------------------------
# State Persistence Tests
# -----------------------------------------------------------------------

class TestStatePersistence:
    """Tests for saving and loading pipeline state."""

    def test_save_and_load_state(self, pipeline):
        pipeline.begin_stage(Stage.RESEARCH)
        pipeline.complete_stage(Stage.RESEARCH, {"bib": "refs.md"})
        pipeline.advance()

        path = pipeline.save_state()
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert data["current_stage"] == Stage.WRITE.value

    def test_round_trip_state(self, pipeline, sample_config):
        pipeline.state.current_stage = Stage.REVISE
        pipeline.state.revision_round = 1
        pipeline.state.integrity_attempts = 2

        path = pipeline.save_state()

        new_pipeline = AcademicPipeline(sample_config)
        new_pipeline.load_state(path)

        assert new_pipeline.state.current_stage == Stage.REVISE
        assert new_pipeline.state.revision_round == 1
        assert new_pipeline.state.integrity_attempts == 2

    def test_stage_result_serializable(self):
        result = StageResult(
            stage=Stage.WRITE,
            status=StageStatus.COMPLETED,
            deliverables={"draft": "draft.md"},
        )
        d = result.to_dict()
        assert d["stage"] == "2_write"
        assert d["status"] == "completed"
        assert "draft" in d["deliverables"]

    def test_pipeline_state_serializable(self):
        state = PipelineState(
            config={"title": "Test"},
            current_stage=Stage.REVIEW,
            review_decision=ReviewDecision.MINOR_REVISION,
        )
        d = state.to_dict()
        assert d["current_stage"] == "3_review"
        assert d["review_decision"] == "minor_revision"


# -----------------------------------------------------------------------
# Quality Gate Integration Tests
# -----------------------------------------------------------------------

class TestQualityGateIntegration:
    """Integration tests combining quality gate with pipeline."""

    def test_pipeline_quality_gate(self, pipeline, sample_paper_text):
        pipeline.quality_gate = QualityGate(
            min_word_count=100,
            max_slop_warnings=50,
            min_burstiness=0.1,
            min_references=5,
        )
        passed, failures = pipeline.run_quality_gate(sample_paper_text)
        assert passed is True

    def test_pipeline_integrity_check(self, pipeline, sample_paper_text):
        check = pipeline.run_integrity_check(sample_paper_text)
        assert check.citations_matched is True
        assert check.references_verified > 0
