"""NPC population orchestration for scripted agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from siege_env.agents.scripted import NPCRole, ScriptedNPCAgent


ROLE_SEQUENCE: tuple[NPCRole, ...] = ("lead", "verifier", "contrarian")


@dataclass(slots=True)
class NPCPopulation:
    """Build and drive deterministic scripted NPC populations."""

    seed: int
    seat_agent_id: int
    total_agents: int = 8
    agents: list[ScriptedNPCAgent] = field(init=False)

    def __post_init__(self) -> None:
        if self.total_agents < 2:
            raise ValueError("total_agents must be at least 2.")
        if self.seat_agent_id < 0 or self.seat_agent_id >= self.total_agents:
            raise ValueError("seat_agent_id must be within [0, total_agents).")
        self.agents = self._build_agents()

    def _build_agents(self) -> list[ScriptedNPCAgent]:
        npc_agents: list[ScriptedNPCAgent] = []
        role_index = 0
        for agent_id in range(self.total_agents):
            if agent_id == self.seat_agent_id:
                continue
            role = ROLE_SEQUENCE[role_index % len(ROLE_SEQUENCE)]
            role_index += 1
            npc_agents.append(
                ScriptedNPCAgent(
                    agent_id=agent_id,
                    role=role,
                    seed=self.seed + agent_id * 17,
                )
            )
        return npc_agents

    def generate_claims(self, template: dict[str, Any], *, step_number: int) -> list[dict[str, Any]]:
        """Generate one claim per scripted NPC in stable agent-id order."""

        return [
            agent.generate_claim(template, step_number=step_number)
            for agent in sorted(self.agents, key=lambda npc: npc.agent_id)
        ]
