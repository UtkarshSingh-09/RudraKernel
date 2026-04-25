"""Unit tests for incident template generator."""

from __future__ import annotations

from siege_env.incidents.generator import generate_variant
from siege_env.incidents.loader import load_templates


class TestIncidentGenerator:
    def test_variant_preserves_root_cause(self) -> None:
        template = load_templates()[0]
        variant = generate_variant(template, variant_index=1)
        assert variant["root_cause"] == template["root_cause"]

    def test_variant_id_includes_index(self) -> None:
        template = load_templates()[0]
        variant = generate_variant(template, variant_index=42)
        assert "42" in variant["id"] or "_v" in variant["id"]

    def test_variant_preserves_signal_count(self) -> None:
        template = load_templates()[0]
        variant = generate_variant(template, variant_index=3)
        assert len(variant["observable_signals"]) == len(template["observable_signals"])

    def test_multiple_variants_from_same_template(self) -> None:
        template = load_templates()[0]
        variants = [generate_variant(template, variant_index=i) for i in range(5)]
        ids = [v["id"] for v in variants]
        assert len(set(ids)) == 5  # all unique IDs
