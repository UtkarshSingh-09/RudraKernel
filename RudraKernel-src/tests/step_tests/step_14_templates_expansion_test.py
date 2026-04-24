"""Gate tests for Step 14 — Incident template expansion to 20."""

from __future__ import annotations

import pytest

from siege_env.incidents.loader import load_templates


EXPECTED_IDS = [
    "gitlab_2017_01_db_recovery",
    "cloudflare_2019_07_regex_waf",
    "aws_s3_2017_02_us_east_1",
    "github_2018_10_network_partition",
    "google_sre_shakespeare_case",
    "slack_2021_01_dns_dependency",
    "meta_2021_10_bgp_withdrawal",
    "fastly_2021_06_edge_config_bug",
    "gcp_2020_11_auth_quota_exhaustion",
    "azure_2023_01_wan_issue",
    "dropbox_2014_01_auth_bug",
    "stripe_2019_07_db_failover",
    "twilio_2023_08_kv_dependency",
    "atlassian_2022_04_script_failure",
    "datadog_2021_11_message_bus",
    "zoom_2020_08_dns_registrar",
    "shopify_2020_09_kubernetes",
    "netflix_2012_12_aws_outage",
    "pagerduty_2023_01_db_migration",
    "openai_2023_11_capacity_event",
]


@pytest.fixture(scope="module")
def expanded_templates() -> list[dict[str, object]]:
    return load_templates(include_step14_expansion=True)


def test_step14_total_template_count(expanded_templates: list[dict[str, object]]) -> None:
    assert len(expanded_templates) == 20


@pytest.mark.parametrize("template_id", EXPECTED_IDS)
def test_step14_template_present_and_valid(
    template_id: str,
    expanded_templates: list[dict[str, object]],
) -> None:
    by_id = {str(t["id"]): t for t in expanded_templates}
    assert template_id in by_id
    template = by_id[template_id]
    assert str(template["source_url"]).startswith("https://")
    assert str(template["root_cause"]).strip()
    assert isinstance(template["observable_signals"], list) and len(template["observable_signals"]) > 0
    assert isinstance(template["flaw_types"], list) and len(template["flaw_types"]) > 0
    assert isinstance(template["blast_radius"], list) and len(template["blast_radius"]) > 0
