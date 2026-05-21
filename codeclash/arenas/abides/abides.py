import json
import shlex
import subprocess

from codeclash.agents.player import Player
from codeclash.arenas.arena import CodeArena, RoundStats
from codeclash.constants import RESULT_TIE
from codeclash.utils.environment import assert_zero_exit_code

RESULTS_JSON = "abides_results.json"
CRASH_SCORE = -1_000_000.0


class ABIDESArena(CodeArena):
    name: str = "ABIDES"
    submission: str = "abides_agent.py"
    description: str = """ABIDES is an agent-based market simulator for financial-market research.

Your bot is a Python file named `abides_agent.py` that defines a class named `MyAgent`.
`MyAgent` should be an ABIDES trading agent class, for example:

    from agent.ValueAgent import ValueAgent as MyAgent

Each round runs several compact ABIDES market simulations. Every submitted agent is evaluated in
identical seeded market worlds with the same exchange, market maker, and background traders. The
objective is to maximize average mark-to-market profit across all simulations in the round.
"""
    default_args: dict = {
        "sims_per_round": 3,
        "market_minutes": 5,
        "background_agents": 3,
        "validation_timeout": 10,
        "player_timeout": 60,
        "timeout": 240,
    }

    def _game_arg(self, key: str):
        return self.game_config.get("args", {}).get(key, self.game_config.get(key, self.default_args[key]))

    def validate_code(self, agent: Player) -> tuple[bool, str | None]:
        quoted_submission = shlex.quote(self.submission)
        file_check = agent.environment.execute(f"test -f {quoted_submission} && echo exists")
        if "exists" not in file_check["output"]:
            return False, f"Submission file `{self.submission}` not found in the workspace root"

        content = agent.environment.execute(f"cat {quoted_submission}")["output"]
        if not content.strip():
            return False, f"`{self.submission}` is empty"

        syntax_check = agent.environment.execute(f"python -m py_compile {quoted_submission}")
        if syntax_check["returncode"] != 0:
            return False, f"Python syntax error in `{self.submission}`:\n{syntax_check['output']}"

        import_check = agent.environment.execute(
            "python - <<'PY'\n"
            "import importlib.util\n"
            "import numpy as np\n"
            "from agent.TradingAgent import TradingAgent\n"
            f"spec = importlib.util.spec_from_file_location('submission_agent', {self.submission!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "assert hasattr(module, 'MyAgent'), 'MyAgent class not found'\n"
            "assert issubclass(module.MyAgent, TradingAgent), 'MyAgent must inherit from an ABIDES TradingAgent class'\n"
            "module.MyAgent(\n"
            "    id=1,\n"
            "    name='validation',\n"
            "    type='ValidationAgent',\n"
            "    symbol='JPM',\n"
            "    starting_cash=10000000,\n"
            "    log_orders=False,\n"
            "    random_state=np.random.RandomState(seed=1),\n"
            ")\n"
            "PY",
            timeout=int(self._game_arg("validation_timeout")),
        )
        if import_check["returncode"] != 0:
            return (
                False,
                f"Could not import and instantiate `MyAgent` from `{self.submission}`:\n{import_check['output']}",
            )

        return True, None

    def execute_round(self, agents: list[Player]) -> None:
        agent_args = []
        for agent in agents:
            agent_args.extend(["--agent", f"{agent.name}=/{agent.name}/{self.submission}"])

        cmd = [
            "python",
            "run_abides.py",
            "--sims",
            str(self._game_arg("sims_per_round")),
            "--market-minutes",
            str(self._game_arg("market_minutes")),
            "--background-agents",
            str(self._game_arg("background_agents")),
            "--player-timeout",
            str(self._game_arg("player_timeout")),
            "--output",
            str(self.log_env / RESULTS_JSON),
            *agent_args,
        ]
        full_cmd = " ".join(shlex.quote(part) for part in cmd)
        self.logger.info(f"Running game: {full_cmd}")
        try:
            response = self.environment.execute(full_cmd, timeout=int(self._game_arg("timeout")))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ABIDES round timed out") from exc
        assert_zero_exit_code(response, logger=self.logger)

    def get_results(self, agents: list[Player], round_num: int, stats: RoundStats):
        result_file = self.log_round(round_num) / RESULTS_JSON
        if not result_file.exists():
            self.logger.error(f"Missing result file: {result_file}")
            stats.winner = RESULT_TIE
            stats.scores = {agent.name: CRASH_SCORE for agent in agents}
            for agent in agents:
                stats.player_stats[agent.name].score = CRASH_SCORE
                stats.details.append(
                    json.dumps(
                        {
                            "player": agent.name,
                            "score": CRASH_SCORE,
                            "status": "error",
                            "error": f"missing ABIDES result file: {result_file}",
                        },
                        sort_keys=True,
                    )
                )
            return

        with open(result_file) as f:
            result = json.load(f)

        scores = {agent.name: 0.0 for agent in agents}
        for player, score in result.get("average_scores", {}).items():
            if player in scores:
                scores[player] = float(score)
        missing_players = sorted(set(scores) - set(result.get("average_scores", {})))
        for player in missing_players:
            scores[player] = CRASH_SCORE
            stats.details.append(
                json.dumps(
                    {
                        "player": player,
                        "score": CRASH_SCORE,
                        "status": "error",
                        "error": "missing ABIDES score",
                    },
                    sort_keys=True,
                )
            )

        stats.scores = scores
        stats.details.extend(result.get("details", []))
        for player, score in scores.items():
            stats.player_stats[player].score = score

        if not scores:
            stats.winner = RESULT_TIE
            return

        top_score = max(scores.values())
        winners = [player for player, score in scores.items() if score == top_score]
        stats.winner = winners[0] if len(winners) == 1 else RESULT_TIE
