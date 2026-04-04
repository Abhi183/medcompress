"""Tests for the stop-slop writing quality filter.

Covers all checker functions: flagged terms, punctuation control,
throat-clearing detection, structure patterns, burstiness analysis,
synonym cycling, and the integrated analyze/report pipeline.
"""

import pytest

from academic_pipeline.stop_slop import (
    BINARY_CONTRAST_LIMIT,
    EM_DASH_LIMIT,
    FLAGGED_TERMS,
    Severity,
    SlopReport,
    Violation,
    analyze,
    check_burstiness,
    check_flagged_terms,
    check_punctuation,
    check_structure_patterns,
    check_synonym_cycling,
    check_throat_clearing,
    format_report,
)


# -----------------------------------------------------------------------
# A. Flagged Term Detection
# -----------------------------------------------------------------------

class TestFlaggedTerms:
    """Tests for high-frequency AI-typical term detection."""

    def test_detects_single_flagged_term(self):
        text = "We delve into the compression techniques."
        violations = check_flagged_terms(text)
        assert len(violations) == 1
        assert violations[0].rule == "flagged_term"
        assert "delve" in violations[0].message.lower()

    def test_detects_multiple_flagged_terms(self):
        text = (
            "This pivotal study will delve into the intricate tapestry "
            "of model compression."
        )
        violations = check_flagged_terms(text)
        terms_found = {v.message.split("'")[1] for v in violations}
        assert "pivotal" in terms_found
        assert "delve" in terms_found
        assert "intricate" in terms_found
        assert "tapestry" in terms_found

    def test_case_insensitive(self):
        text = "The PIVOTAL role of Quantization."
        violations = check_flagged_terms(text)
        assert len(violations) >= 1

    def test_exempt_technical_usage(self):
        """'robust' in statistics context should be exempt."""
        text = "We used a robust estimator for the regression analysis."
        violations = check_flagged_terms(text)
        robust_violations = [v for v in violations if "robust" in v.message.lower()]
        assert len(robust_violations) == 0

    def test_non_exempt_usage(self):
        """'robust' outside statistics context should be flagged."""
        text = "This provides a robust framework for compression."
        violations = check_flagged_terms(text)
        robust_violations = [v for v in violations if "robust" in v.message.lower()]
        assert len(robust_violations) == 1

    def test_clean_text_no_violations(self):
        text = (
            "Model compression reduces neural network size "
            "while preserving accuracy. Quantization converts "
            "floating-point weights to lower-precision integers."
        )
        violations = check_flagged_terms(text)
        assert len(violations) == 0

    def test_violation_has_alternatives(self):
        text = "We leverage transfer learning for compression."
        violations = check_flagged_terms(text)
        assert len(violations) >= 1
        assert "use" in violations[0].suggestion.lower()

    def test_provides_line_and_column(self):
        text = "Line one.\nWe delve into the topic."
        violations = check_flagged_terms(text)
        assert len(violations) >= 1
        assert violations[0].line == 2
        assert violations[0].column > 0


# -----------------------------------------------------------------------
# B. Punctuation Control
# -----------------------------------------------------------------------

class TestPunctuation:
    """Tests for em-dash and semicolon overuse."""

    def test_em_dash_within_limit(self):
        text = "First clause — second clause. That is all."
        violations = check_punctuation(text)
        em_violations = [v for v in violations if v.rule == "em_dash_overuse"]
        assert len(em_violations) == 0

    def test_em_dash_over_limit(self):
        dashes = " — ".join(["clause"] * (EM_DASH_LIMIT + 2))
        text = f"The {dashes} end."
        violations = check_punctuation(text)
        em_violations = [v for v in violations if v.rule == "em_dash_overuse"]
        assert len(em_violations) > 0

    def test_detects_double_dash(self):
        text = "First -- second -- third -- fourth -- fifth end."
        violations = check_punctuation(text)
        em_violations = [v for v in violations if v.rule == "em_dash_overuse"]
        assert len(em_violations) > 0

    def test_semicolons_within_limit(self):
        text = "First point; second point. That is the full story."
        violations = check_punctuation(text)
        semi_violations = [v for v in violations if v.rule == "semicolon_overuse"]
        assert len(semi_violations) == 0

    def test_semicolons_over_limit_for_short_text(self):
        # Short text (~50 words) with many semicolons
        text = (
            "Point one; point two; point three; point four; "
            "point five; point six; point seven; done."
        )
        violations = check_punctuation(text)
        semi_violations = [v for v in violations if v.rule == "semicolon_overuse"]
        assert len(semi_violations) > 0


# -----------------------------------------------------------------------
# C. Throat-Clearing Openers
# -----------------------------------------------------------------------

class TestThroatClearing:
    """Tests for throat-clearing openers and meta-commentary."""

    def test_detects_in_the_realm_of(self):
        text = "In the realm of medical imaging, compression is important."
        violations = check_throat_clearing(text)
        assert len(violations) >= 1
        assert "throat_clearing" in violations[0].rule

    def test_detects_in_order_to(self):
        text = "In order to compress the model, we applied QAT."
        violations = check_throat_clearing(text)
        assert len(violations) >= 1

    def test_detects_meta_commentary(self):
        text = "This section will discuss the compression results."
        violations = check_throat_clearing(text)
        assert len(violations) >= 1
        assert "meta-commentary" in violations[0].suggestion.lower()

    def test_does_not_flag_clean_text(self):
        text = (
            "QAT reduces model size by inserting fake quantization "
            "nodes during training. The resulting INT8 model achieves "
            "comparable accuracy to the FP32 baseline."
        )
        violations = check_throat_clearing(text)
        assert len(violations) == 0

    def test_detects_it_should_be_noted(self):
        text = "It should be noted that the model size decreased by 4x."
        violations = check_throat_clearing(text)
        assert len(violations) >= 1

    def test_detects_it_goes_without_saying(self):
        text = "It goes without saying that accuracy matters in healthcare."
        violations = check_throat_clearing(text)
        assert len(violations) >= 1


# -----------------------------------------------------------------------
# D. Structure Patterns
# -----------------------------------------------------------------------

class TestStructurePatterns:
    """Tests for structural monotony and binary contrast overuse."""

    def test_binary_contrast_within_limit(self):
        text = (
            "It is not about speed. It is about accuracy. "
            "The model performs well on both metrics."
        )
        violations = check_structure_patterns(text)
        contrast_v = [v for v in violations
                      if v.rule == "binary_contrast_overuse"]
        assert len(contrast_v) == 0

    def test_uniform_paragraph_length_detected(self):
        # 6 paragraphs all approximately the same length
        para = "Word " * 50
        text = "\n\n".join([para.strip()] * 6)
        violations = check_structure_patterns(text)
        uniform_v = [v for v in violations
                     if v.rule == "uniform_paragraph_length"]
        assert len(uniform_v) >= 1

    def test_varied_paragraphs_pass(self):
        text = (
            "Short paragraph here.\n\n"
            + "Medium length paragraph with more words in it for variation. " * 3
            + "\n\n"
            + "Another short one.\n\n"
            + "A much longer paragraph that goes on and on with many words "
            "to demonstrate significant variation in paragraph length across "
            "the document so the checker does not flag it. " * 4
        )
        violations = check_structure_patterns(text)
        uniform_v = [v for v in violations
                     if v.rule == "uniform_paragraph_length"]
        assert len(uniform_v) == 0


# -----------------------------------------------------------------------
# E. Burstiness (Sentence Length Variation)
# -----------------------------------------------------------------------

class TestBurstiness:
    """Tests for sentence length variation."""

    def test_monotonous_sentences_flagged(self):
        # 8 sentences all exactly the same length (~20 words each)
        sentence = "The model achieves good accuracy on the test set for this task. "
        text = sentence * 8
        violations = check_burstiness(text)
        assert len(violations) >= 1
        assert violations[0].rule == "low_burstiness"

    def test_varied_sentences_pass(self):
        text = (
            "Short sentence. "
            "The model achieves remarkable accuracy on the ISIC test set. "
            "It works. "
            "Knowledge distillation transfers learned representations from "
            "the larger teacher model to the smaller student architecture "
            "through soft probability matching at elevated temperature. "
            "QAT helps too. "
            "The results demonstrate the viability of compressed medical "
            "imaging models for mobile deployment scenarios."
        )
        violations = check_burstiness(text)
        assert len(violations) == 0

    def test_too_few_sentences_skipped(self):
        text = "One sentence. Two. Three."
        violations = check_burstiness(text)
        assert len(violations) == 0


# -----------------------------------------------------------------------
# F. Synonym Cycling
# -----------------------------------------------------------------------

class TestSynonymCycling:
    """Tests for synonym cycling detection within paragraphs."""

    def test_detects_cycling(self):
        text = (
            "The method uses a novel approach. This technique reduces "
            "model size. The strategy achieves good compression ratios."
        )
        violations = check_synonym_cycling(text)
        cycling_v = [v for v in violations if v.rule == "synonym_cycling"]
        assert len(cycling_v) >= 1

    def test_consistent_terminology_passes(self):
        text = (
            "The method compresses the model. This method reduces "
            "the parameter count. The method achieves a 4x size reduction."
        )
        violations = check_synonym_cycling(text)
        cycling_v = [v for v in violations if v.rule == "synonym_cycling"]
        assert len(cycling_v) == 0

    def test_cycling_across_paragraphs_not_flagged(self):
        """Synonym cycling is per-paragraph, not per-document."""
        text = (
            "The method is effective.\n\n"
            "This approach works well.\n\n"
            "The technique is novel."
        )
        violations = check_synonym_cycling(text)
        cycling_v = [v for v in violations if v.rule == "synonym_cycling"]
        assert len(cycling_v) == 0


# -----------------------------------------------------------------------
# G. Integrated Analysis
# -----------------------------------------------------------------------

class TestAnalyze:
    """Tests for the integrated analyze() function."""

    def test_clean_text_produces_clean_report(self):
        text = (
            "Model compression reduces neural network size while "
            "preserving diagnostic accuracy. Quantization converts "
            "floating-point weights to lower-precision integers. "
            "This enables deployment on resource-constrained devices.\n\n"
            "The ISIC 2020 dataset provides 33,126 dermoscopy images. "
            "We split the data using stratified sampling."
        )
        report = analyze(text)
        assert isinstance(report, SlopReport)
        assert report.word_count > 0
        assert report.sentence_count > 0
        assert report.paragraph_count == 2

    def test_report_counts_em_dashes(self):
        text = "Clause one — clause two — clause three — clause four."
        report = analyze(text)
        assert report.em_dash_count == 3

    def test_burstiness_score_range(self):
        text = (
            "Short. "
            "A much longer sentence with many words for variation. "
            "Medium one. "
            "Another long sentence that provides good burstiness in "
            "the analysis of sentence length variation across the text."
        )
        report = analyze(text)
        assert 0.0 <= report.burstiness_score <= 1.0

    def test_violations_sorted_by_line(self):
        text = (
            "In the realm of compression.\n"
            "In order to improve accuracy.\n"
            "We delve into the results."
        )
        report = analyze(text)
        lines = [v.line for v in report.violations]
        assert lines == sorted(lines)

    def test_severity_counts(self):
        text = "We delve into the pivotal landscape of intricate compression."
        report = analyze(text)
        assert report.warning_count >= 3  # at least delve, pivotal, intricate
        assert report.error_count == 0

    def test_section_name_accepted(self):
        text = "Results show 4x compression with 2% accuracy loss."
        report = analyze(text, section_name="results")
        assert isinstance(report, SlopReport)


# -----------------------------------------------------------------------
# H. Report Formatting
# -----------------------------------------------------------------------

class TestFormatReport:
    """Tests for the human-readable report formatter."""

    def test_clean_report_format(self):
        text = "Model compression reduces size while preserving accuracy."
        report = analyze(text)
        output = format_report(report)
        assert "STOP-SLOP" in output
        assert "Words:" in output

    def test_violation_report_includes_details(self):
        text = "We delve into the pivotal compression landscape."
        report = analyze(text)
        output = format_report(report)
        assert "violation" in output.lower()
        assert "delve" in output.lower()
        assert "→" in output  # suggestion arrow

    def test_report_shows_burstiness(self):
        text = "A sentence. Another sentence. Third sentence."
        report = analyze(text)
        output = format_report(report)
        assert "Burstiness" in output
