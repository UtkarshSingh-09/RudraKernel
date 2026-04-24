from __future__ import annotations

from siege_env.agents.population import NPCPopulation
from siege_env.agents.scripted import ROLE_CONFIDENCE_BOUNDS, ScriptedNPCAgent
from siege_env.incidents.loader import load_templates
from siege_env.server.siege_environment import SIEGEEnvironment


def _template() -> dict[str, object]:
    return load_templates()[0]


def test_population_builds_seven_npcs_for_one_seat() -> None:
    population = NPCPopulation(seed=11, seat_agent_id=0)
    agents = population.agents
    assert len(agents) == 7
    assert all(agent.agent_id != 0 for agent in agents)


def test_population_role_assignment_is_deterministic() -> None:
    first = NPCPopulation(seed=11, seat_agent_id=0)
    second = NPCPopulation(seed=11, seat_agent_id=0)
    first_roles = [agent.role for agent in first.agents]
    second_roles = [agent.role for agent in second.agents]
    assert first_roles == second_roles


def test_scripted_agent_generate_claim_is_deterministic() -> None:
    template = _template()
    agent_a = ScriptedNPCAgent(agent_id=3, role="verifier", seed=91)
    agent_b = ScriptedNPCAgent(agent_id=3, role="verifier", seed=91)
    assert agent_a.generate_claim(template, step_number=2) == agent_b.generate_claim(
        template, step_number=2
    )


def test_claim_contains_expected_keys() -> None:
    template = _template()
    claim = ScriptedNPCAgent(agent_id=4, role="lead", seed=99).generate_claim(template, step_number=1)
    assert {"agent_id", "claim_id", "root_cause", "confidence", "role", "evidence"} <= set(claim.keys())


def test_role_confidence_respects_role_bounds() -> None:
    template = _template()
    for role, bounds in ROLE_CONFIDENCE_BOUNDS.items():
        claim = ScriptedNPCAgent(agent_id=2, role=role, seed=7).generate_claim(template, step_number=1)
        lower, upper = bounds
        assert lower <= claim["confidence"] <= upper


def test_population_generate_claims_returns_one_per_npc() -> None:
    population = NPCPopulation(seed=13, seat_agent_id=0)
    claims = population.generate_claims(_template(), step_number=2)
    assert len(claims) == len(population.agents)
    assert len({claim["agent_id"] for claim in claims}) == len(claims)


def test_population_claim_generation_is_deterministic() -> None:
    template = _template()
    pop_a = NPCPopulation(seed=19, seat_agent_id=0)
    pop_b = NPCPopulation(seed=19, seat_agent_id=0)
    assert pop_a.generate_claims(template, step_number=3) == pop_b.generate_claims(
        template, step_number=3
    )


def test_environment_observation_contains_npc_claims() -> None:
    env = SIEGEEnvironment(seed=21)
    obs = env.reset()
    assert len(obs.agent_claims) == 7
