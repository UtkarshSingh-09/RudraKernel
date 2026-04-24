from __future__ import annotations

from siege_env.incidents.generator import generate_variant
from siege_env.incidents.loader import load_templates


REQUIRED_KEYS = {
    "id",
    "source_url",
    "root_cause",
    "observable_signals",
    "flaw_types",
    "blast_radius",
}


def test_all_five_seed_templates_load() -> None:
    templates = load_templates()
    assert len(templates) == 5


def test_templates_contain_required_ground_truth_fields() -> None:
    templates = load_templates()
    for template in templates:
        assert REQUIRED_KEYS.issubset(set(template.keys()))
        assert template["id"]
        assert template["root_cause"]
        assert template["source_url"].startswith("https://")
        assert isinstance(template["observable_signals"], list)
        assert isinstance(template["flaw_types"], list)
        assert isinstance(template["blast_radius"], list)


def test_variant_generator_produces_valid_variant() -> None:
    template = load_templates()[0]
    variant = generate_variant(template, variant_index=7)
    assert REQUIRED_KEYS.issubset(set(variant.keys()))
    assert variant["id"].startswith(f"{template['id']}_v")
    assert variant["root_cause"] == template["root_cause"]
    assert variant["source_url"] == template["source_url"]
    assert len(variant["observable_signals"]) == len(template["observable_signals"])
