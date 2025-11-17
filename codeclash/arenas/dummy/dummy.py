import re

from codeclash.agents.player import Player
from codeclash.arenas.arena import CodeArena, RoundStats
from codeclash.utils.environment import assert_zero_exit_code

DUMMY_LOG = "result.log"


class DummyArena(CodeArena):
    name: str = "Dummy"
    description: str = """WARNING: This is a dummy game meant for testing the CodeClash infrastructure. It does not represent a real game."""
    submission: str = "main.py"

    def execute_round(self, agents: list[Player]) -> None:
        args = [f"/{agent.name}/{self.submission}" for agent in agents]
        cmd = f"python engine.py {' '.join(args)} -r {self.game_config['sims_per_round']} > {self.log_env / DUMMY_LOG};"
        self.logger.info(f"Running game: {cmd}")
        assert_zero_exit_code(self.environment.execute(cmd))

    def get_results(self, agents: list[Player], round_num: int, stats: RoundStats):
        with open(self.log_round(round_num) / DUMMY_LOG) as f:
            round_log = f.read()
        lines = round_log.split("FINAL_RESULTS")[-1].splitlines()

        scores = {}
        for line in lines:
            match = re.search(r"Bot\_(\d)\_main:\s(\d+)\srounds\swon", line)
            if match:
                bot_id = match.group(1)
                rounds_won = int(match.group(2))
                scores[agents[int(bot_id) - 1].name] = rounds_won

        stats.winner = max(scores, key=scores.get) if scores else "unknown"
        stats.scores = scores
        for player, score in scores.items():
            stats.player_stats[player].score = score

    def validate_code(self, agent: Player) -> tuple[bool, str | None]:
        # Check that the submission file exists
        file_check = agent.environment.execute(f"test -f {self.submission} && echo 'exists'")
        if "exists" not in file_check["output"]:
            return False, f"Submission file '{self.submission}' not found"

        # Check that the file is not empty
        file_content = agent.environment.execute(f"cat {self.submission}")["output"]
        if not file_content.strip():
            return False, f"Submission file '{self.submission}' is empty"

        # Validate Python syntax
        syntax_check = agent.environment.execute(f"python -m py_compile {self.submission}")
        if syntax_check["returncode"] != 0:
            return False, f"Python syntax error in '{self.submission}':\n{syntax_check['output']}"

        return True, None
