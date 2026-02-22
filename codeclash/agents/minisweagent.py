import logging
import os
import random
import time
import traceback
from collections.abc import Callable

from minisweagent import Model
from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.models.test_models import DeterministicModel

try:
    # mini-swe-agent v1 path
    from minisweagent.run.utils.save import save_traj as _save_traj_impl
except ModuleNotFoundError:
    _save_traj_impl = None

from codeclash import REPO_DIR
from codeclash.agents.player import Player
from codeclash.agents.utils import GameContext
from codeclash.utils.environment import copy_to_container

os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "90")
os.environ["LITELLM_MODEL_REGISTRY_PATH"] = str(
    (REPO_DIR / "configs" / "mini" / "litellm_custom_model_config.yaml").resolve()
)


def _save_traj_compat(
    agent: DefaultAgent,
    traj_path,
    *,
    exit_status: str | None,
    result: str | None,
    print_fct: Callable | None = None,
) -> None:
    if _save_traj_impl is not None:
        _save_traj_impl(
            agent,
            traj_path,
            exit_status=exit_status,
            result=result,
            print_fct=print_fct,
        )
        return

    # mini-swe-agent v2 removed run.utils.save; fall back to agent serialization.
    agent.save(
        traj_path,
        {
            "info": {
                "exit_status": exit_status or "",
                "result": result or "",
            }
        },
    )
    if print_fct is not None:
        print_fct("Saved trajectory via compatibility fallback (mini-swe-agent v2).")


class ClashAgent(DefaultAgent):
    """
    Slightly modified version of `DefaultAgent` from mini-SWE-agent
    (https://github.com/SWE-agent/mini-swe-agent)
    """

    def __init__(
        self,
        model: Model,
        env: DockerEnvironment,
        *,
        logger: logging.Logger,
        config_class: Callable = AgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.logger = logger

    def add_message(self, role: str, content: str, **kwargs):
        super().add_message(role, content, **kwargs)
        self.logger.debug(f"[{role}] {content}", extra={"highlighter": None})

    def query(self) -> dict:
        """Query model with provider-safe messages.

        mini-swe-agent stores per-message `timestamp` fields for trajectories.
        Some providers (including Groq OpenAI-compatible endpoint) reject unknown
        keys in message objects, so we always send canonical role/content payloads.
        """
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        provider_messages = [{"role": message["role"], "content": str(message.get("content", ""))} for message in self.messages]
        max_attempts = int(os.getenv("MSWEA_PROVIDER_RETRY_ATTEMPTS", "3"))
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.model.query(provider_messages)
                break
            except Exception as e:
                message = str(e).lower()
                is_flex_capacity_error = "capacity_exceeded" in message or " 498" in message or "status 498" in message
                if not is_flex_capacity_error or attempt == max_attempts:
                    raise
                # Jittered backoff for Groq Flex transient capacity misses.
                sleep_s = min(2**attempt, 16) + random.uniform(0.0, 0.75)
                self.logger.warning(
                    "Transient provider capacity error (attempt %s/%s). Retrying in %.2fs.",
                    attempt,
                    max_attempts,
                    sleep_s,
                )
                time.sleep(sleep_s)
        assert response is not None
        self.add_message("assistant", **response)
        return response


class MiniSWEAgent(Player):
    """Player with agentic code editing capabilities"""

    def __init__(self, config: dict, environment: DockerEnvironment, game_context: GameContext):
        super().__init__(config, environment=environment, game_context=game_context)

    def run(self):
        # temporary workaround around https://github.com/SWE-agent/mini-swe-agent/issues/477
        if "DeterministicModel" not in self.config["config"]["model"].get("model_class", ""):
            model = get_model(config=self.config["config"]["model"])
        else:
            model = DeterministicModel(outputs=self.config["config"]["model"]["outputs"])
        self.agent = ClashAgent(
            model=model,
            env=self.environment,
            logger=self.logger,
            **self.config["config"]["agent"],
        )
        exit_status = None
        result = None
        exc_message = None
        try:
            exit_status, result = self.agent.run(task="", **self.game_context.to_template_vars())
        except Exception as e:
            exit_status = str(e)
            exc_message = traceback.format_exc()
            result = exc_message
            self.logger.critical(exc_message)
        finally:
            traj_path = (
                self.game_context.log_local
                / "players"
                / self.name
                / f"{self.name}_r{self.game_context.round}.traj.json"
            )
            _save_traj_compat(
                self.agent,  # type: ignore
                traj_path,
                exit_status=exit_status,
                result=result,
                print_fct=self.logger.debug,
            )
            copy_to_container(
                self.environment,
                traj_path,
                self.game_context.log_env / "edits" / traj_path.name,
            )
            self._metadata["agent_stats"][self.game_context.round] = {
                "exit_status": exit_status,
                "cost": self.agent.model.cost,
                "api_calls": self.agent.model.n_calls,
            }
        if exit_status.lower().strip() not in ["", "submitted", "limitsexceeded"] and exc_message is not None:
            raise RuntimeError(f"Agent {self.name} failed with exit status: {exit_status} and exception: {exc_message}")
