import json
from pathlib import Path

from codeclash.arenas.abides.abides import CRASH_SCORE, ABIDESArena
from codeclash.arenas.arena import RoundStats
from codeclash.constants import RESULT_TIE

from .conftest import MockEnvironment, MockPlayer


class TestABIDESValidation:
    def test_valid_agent(self, mock_player_factory):
        arena = ABIDESArena.__new__(ABIDESArena)
        arena.submission = "abides_agent.py"
        player = mock_player_factory(
            name="Alice",
            files={"abides_agent.py": "from agent.ValueAgent import ValueAgent as MyAgent\n"},
            command_outputs={
                "test -f abides_agent.py && echo exists": {"output": "exists\n", "returncode": 0},
                "cat abides_agent.py": {
                    "output": "from agent.ValueAgent import ValueAgent as MyAgent\n",
                    "returncode": 0,
                },
                "python -m py_compile abides_agent.py": {"output": "", "returncode": 0},
                "python - <<'PY'": {"output": "", "returncode": 0},
            },
        )

        valid, error = arena.validate_code(player)

        assert valid is True
        assert error is None
        import_command = next(cmd for cmd in player.environment._executed_commands if cmd.startswith("python - <<'PY'"))
        assert import_command.index("from agent.TradingAgent import TradingAgent") < import_command.index(
            "spec.loader.exec_module(module)"
        )

    def test_missing_myagent(self, mock_player_factory):
        arena = ABIDESArena.__new__(ABIDESArena)
        arena.submission = "abides_agent.py"
        player = mock_player_factory(
            name="Alice",
            files={"abides_agent.py": "class OtherAgent:\n    pass\n"},
            command_outputs={
                "test -f abides_agent.py && echo exists": {"output": "exists\n", "returncode": 0},
                "cat abides_agent.py": {"output": "class OtherAgent:\n    pass\n", "returncode": 0},
                "python -m py_compile abides_agent.py": {"output": "", "returncode": 0},
                "python - <<'PY'": {"output": "MyAgent class not found", "returncode": 1},
            },
        )

        valid, error = arena.validate_code(player)

        assert valid is False
        assert "Could not import and instantiate" in error

    def test_import_failure(self, mock_player_factory):
        arena = ABIDESArena.__new__(ABIDESArena)
        arena.submission = "abides_agent.py"
        player = mock_player_factory(
            name="Alice",
            files={"abides_agent.py": "class MyAgent:\n    pass\n"},
            command_outputs={
                "test -f abides_agent.py && echo exists": {"output": "exists\n", "returncode": 0},
                "cat abides_agent.py": {"output": "class MyAgent:\n    pass\n", "returncode": 0},
                "python -m py_compile abides_agent.py": {"output": "", "returncode": 0},
                "python - <<'PY'": {"output": "ImportError", "returncode": 1},
            },
        )

        valid, error = arena.validate_code(player)

        assert valid is False
        assert "Could not import and instantiate" in error


class TestABIDESResults:
    def test_parse_winner(self, tmp_log_dir):
        arena = ABIDESArena.__new__(ABIDESArena)
        arena.log_local = tmp_log_dir
        arena.logger = type("Logger", (), {"error": lambda self, msg: None})()
        round_dir = tmp_log_dir / "rounds" / "1"
        round_dir.mkdir(parents=True)
        (round_dir / "abides_results.json").write_text(
            json.dumps(
                {
                    "average_scores": {"Alice": 125.0, "Bob": 75.0},
                    "details": ['{"sim": 0, "player": "Alice", "score": 125.0}'],
                }
            )
        )

        agents = [MockPlayer("Alice"), MockPlayer("Bob")]
        stats = RoundStats(round_num=1, agents=agents)

        arena.get_results(agents, 1, stats)

        assert stats.winner == "Alice"
        assert stats.scores == {"Alice": 125.0, "Bob": 75.0}
        assert stats.player_stats["Alice"].score == 125.0
        assert stats.details == ['{"sim": 0, "player": "Alice", "score": 125.0}']

    def test_parse_tie(self, tmp_log_dir):
        arena = ABIDESArena.__new__(ABIDESArena)
        arena.log_local = tmp_log_dir
        arena.logger = type("Logger", (), {"error": lambda self, msg: None})()
        round_dir = tmp_log_dir / "rounds" / "1"
        round_dir.mkdir(parents=True)
        (round_dir / "abides_results.json").write_text(json.dumps({"average_scores": {"Alice": 1, "Bob": 1}}))

        agents = [MockPlayer("Alice"), MockPlayer("Bob")]
        stats = RoundStats(round_num=1, agents=agents)

        arena.get_results(agents, 1, stats)

        assert stats.winner == RESULT_TIE
        assert stats.scores == {"Alice": 1.0, "Bob": 1.0}

    def test_missing_score_is_penalized(self, tmp_log_dir):
        arena = ABIDESArena.__new__(ABIDESArena)
        arena.log_local = tmp_log_dir
        arena.logger = type("Logger", (), {"error": lambda self, msg: None})()
        round_dir = tmp_log_dir / "rounds" / "1"
        round_dir.mkdir(parents=True)
        (round_dir / "abides_results.json").write_text(json.dumps({"average_scores": {"Bob": -5.0}}))

        agents = [MockPlayer("Alice"), MockPlayer("Bob")]
        stats = RoundStats(round_num=1, agents=agents)

        arena.get_results(agents, 1, stats)

        assert stats.winner == "Bob"
        assert stats.scores == {"Alice": CRASH_SCORE, "Bob": -5.0}
        assert "missing ABIDES score" in stats.details[0]


class TestABIDESExecution:
    def test_execute_round_uses_nested_game_args(self):
        arena = ABIDESArena.__new__(ABIDESArena)
        arena.submission = "abides_agent.py"
        arena.config = {
            "game": {
                "sims_per_round": 5,
                "args": {
                    "market_minutes": 11,
                    "background_agents": 13,
                    "timeout": 17,
                },
            }
        }
        arena.log_env = Path("/logs")
        arena.logger = type("Logger", (), {"info": lambda self, msg: None, "error": lambda self, msg: None})()

        class CapturingEnvironment(MockEnvironment):
            def __init__(self):
                super().__init__()
                self.timeout = None

            def execute(self, cmd, cwd=None, timeout=None):
                self._executed_commands.append(cmd)
                self.timeout = timeout
                return {"output": "", "returncode": 0}

        arena.environment = CapturingEnvironment()

        arena.execute_round([MockPlayer("Alice"), MockPlayer("Bob")])

        cmd = arena.environment._executed_commands[0]
        assert "--sims 5" in cmd
        assert "--market-minutes 11" in cmd
        assert "--background-agents 13" in cmd
        assert "--output /logs/abides_results.json" in cmd
        assert "--agent Alice=/Alice/abides_agent.py" in cmd
        assert "--agent Bob=/Bob/abides_agent.py" in cmd
        assert arena.environment.timeout == 17
