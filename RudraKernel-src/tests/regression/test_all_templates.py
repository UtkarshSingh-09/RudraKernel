"""Regression test: validate all templates (seed + expansion + clinical)."""

from __future__ import annotations

import pytest

from siege_env.incidents.loader import load_templates

REQUIRED_KEYS = {"id", "source_url", "root_cause", "observable_signals", "flaw_types", "blast_radius"}


@pytest.fixture(scope="module")
def all_templates() -> list[dict]:
    return load_templates(include_step14_expansion=True)


def test_at_least_20_templates(all_templates: list[dict]) -> None:
    assert len(all_templates) >= 20


def test_no_duplicate_ids(all_templates: list[dict]) -> None:
    ids = [t["id"] for t in all_templates]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"


@pytest.mark.parametrize("idx", range(20))
def test_template_has_required_keys(idx: int, all_templates: list[dict]) -> None:
    if idx >= len(all_templates):
        pytest.skip(f"Template index {idx} out of range")
    template = all_templates[idx]
    missing = REQUIRED_KEYS - set(template.keys())
    assert not missing, f"Template '{template.get('id')}' missing keys: {missing}"


@pytest.mark.parametrize("idx", range(20))
def test_template_signals_non_empty(idx: int, all_templates: list[dict]) -> None:
    if idx >= len(all_templates):
        pytest.skip(f"Template index {idx} out of range")
    template = all_templates[idx]
    assert len(template["observable_signals"]) > 0
    assert len(template["flaw_types"]) > 0
    assert len(template["blast_radius"]) > 0


@pytest.mark.parametrize("idx", range(20))
def test_template_source_url_valid(idx: int, all_templates: list[dict]) -> None:
    if idx >= len(all_templates):
        pytest.skip(f"Template index {idx} out of range")
    template = all_templates[idx]
    assert template["source_url"].startswith("https://")
