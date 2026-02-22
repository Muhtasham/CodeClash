from __future__ import annotations

from typing import TYPE_CHECKING

from minisweagent.environments.docker import DockerEnvironment

from codeclash.agents.dummy_agent import Dummy
from codeclash.agents.player import Player
from codeclash.agents.utils import GameContext

if TYPE_CHECKING:
    from codeclash.agents.minisweagent import MiniSWEAgent


def get_agent(config: dict, game_context: GameContext, environment: DockerEnvironment) -> Player:
    agent_name = config["agent"]
    if agent_name == "dummy":
        return Dummy(config, environment, game_context)
    if agent_name == "mini":
        try:
            from codeclash.agents.minisweagent import MiniSWEAgent
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "mini agent requires optional mini-swe-agent runtime dependencies "
                "(including trajectory save utilities)."
            ) from exc
        return MiniSWEAgent(config, environment, game_context)
    raise ValueError(f"Unknown agent type: {agent_name}")
