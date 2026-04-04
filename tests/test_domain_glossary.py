"""Tests for the MedCompress domain glossary."""

import pytest

from academic_pipeline.domain_glossary import (
    GLOSSARY,
    GlossaryEntry,
    get_abbreviation_map,
    get_all_canonical_terms,
    get_term,
)


class TestGlossaryLookup:
    """Tests for glossary term lookup."""

    def test_get_known_term(self):
        entry = get_term("qat")
        assert entry.canonical == "quantization-aware training"
        assert entry.abbreviation == "QAT"

    def test_get_term_case_insensitive(self):
        entry = get_term("QAT")
        assert entry.canonical == "quantization-aware training"

    def test_unknown_term_raises(self):
        with pytest.raises(KeyError, match="Unknown glossary term"):
            get_term("nonexistent_term")

    def test_entry_has_definition(self):
        for key, entry in GLOSSARY.items():
            assert len(entry.definition) > 10, (
                f"Term '{key}' has too short a definition"
            )

    def test_entry_has_abbreviation(self):
        for key, entry in GLOSSARY.items():
            assert len(entry.abbreviation) > 0, (
                f"Term '{key}' missing abbreviation"
            )


class TestGlossaryCompleteness:
    """Tests for glossary coverage of MedCompress domain."""

    REQUIRED_TERMS = ["qat", "kd", "tflite", "onnx", "dice", "auc",
                      "isic", "brats", "2.5d", "wasm"]

    def test_all_required_terms_present(self):
        for term in self.REQUIRED_TERMS:
            assert term in GLOSSARY, f"Missing required term: {term}"

    def test_no_duplicate_abbreviations(self):
        abbrevs = [e.abbreviation for e in GLOSSARY.values()]
        assert len(abbrevs) == len(set(abbrevs)), "Duplicate abbreviations found"


class TestGlossaryHelpers:
    """Tests for glossary helper functions."""

    def test_get_all_canonical_terms(self):
        terms = get_all_canonical_terms()
        assert len(terms) == len(GLOSSARY)
        assert "quantization-aware training" in terms

    def test_get_abbreviation_map(self):
        abbrev_map = get_abbreviation_map()
        assert abbrev_map["QAT"] == "quantization-aware training"
        assert abbrev_map["KD"] == "knowledge distillation"
        assert abbrev_map["TFLite"] == "TensorFlow Lite"

    def test_entry_is_frozen(self):
        entry = get_term("qat")
        with pytest.raises(AttributeError):
            entry.canonical = "changed"
