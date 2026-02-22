from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path


def _load_robotrumble_prime_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "environments" / "robotrumble_prime" / "robotrumble_prime.py"
    spec = importlib.util.spec_from_file_location("robotrumble_prime", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_robotrumble_prime_canonical_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "environments" / "robotrumble_prime" / "robotrumble_prime_canonical.py"
    spec = importlib.util.spec_from_file_location("robotrumble_prime_canonical", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rrp = _load_robotrumble_prime_module()
rrpc = _load_robotrumble_prime_canonical_module()


def _with_env(overrides: dict[str, str | None], fn):
    original = {k: os.environ.get(k) for k in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return fn()
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_contract_gate_accepts_exact_action_api() -> None:
    code = """
def robot(state, unit):
    direction = Direction.East
    if state.turn % 2 == 0:
        return Action.move(direction)
    return Action.attack(Direction.South)
"""
    assert rrp._has_robot_signature(code)
    assert rrp._contract_gate(code)
    assert rrp._returns_action(code) == 1.0


def test_contract_gate_rejects_non_contract_return() -> None:
    code = """
def robot(state, unit):
    return Action.wait
"""
    assert not rrp._contract_gate(code)
    assert rrp._returns_action(code) == 0.0


def test_contract_gate_rejects_action_redefinition() -> None:
    code = """
class Action:
    pass

def robot(state, unit):
    return Action.move(Direction.East)
"""
    assert not rrp._contract_gate(code)
    assert rrp._returns_action(code) == 0.0


def test_contract_gate_rejects_unknown_direction_source() -> None:
    code = """
def robot(state, unit):
    direction = pick_direction(state)
    return Action.move(direction)
"""
    assert not rrp._contract_gate(code)
    assert rrp._returns_action(code) == 0.0


def test_contract_gate_accepts_direction_to_alias() -> None:
    code = """
def robot(state, unit):
    enemies = state.objs_by_team(state.other_team)
    if not enemies:
        return Action.move(Direction.East)
    closest_enemy = min(enemies, key=lambda e: e.coords.distance_to(unit.coords))
    direction = unit.coords.direction_to(closest_enemy.coords)
    if unit.coords.distance_to(closest_enemy.coords) <= 1:
        return Action.attack(direction)
    return Action.move(direction)
"""
    assert rrp._contract_gate(code)
    assert rrp._returns_action(code) == 1.0


def test_contract_gate_accepts_direction_to_direct_call() -> None:
    code = """
def robot(state, unit):
    enemies = state.objs_by_team(state.other_team)
    if not enemies:
        return Action.move(Direction.West)
    closest_enemy = min(enemies, key=lambda e: e.coords.distance_to(unit.coords))
    if unit.coords.distance_to(closest_enemy.coords) <= 1:
        return Action.attack(unit.coords.direction_to(closest_enemy.coords))
    return Action.move(unit.coords.direction_to(closest_enemy.coords))
"""
    assert rrp._contract_gate(code)
    assert rrp._returns_action(code) == 1.0


def test_signature_rejects_extra_args() -> None:
    code = """
def robot(state, unit, extra):
    return Action.move(Direction.East)
"""
    assert not rrp._has_robot_signature(code)
    assert not rrp._contract_gate(code)


def test_prompt_contract_injected_into_dataset_questions() -> None:
    ds = rrp._build_dataset(split="train", max_examples=1, seed=1337)
    row = ds[0]
    question = row["question"]
    assert "API contract requirements" in question
    assert "Action.move(Direction.X)" in question
    assert "Action.attack(Direction.X)" in question


def test_think_strip_and_malformed_reject() -> None:
    clean = rrp._strip_or_reject_think_blocks(
        "<think>hidden</think>\n\ndef robot(state, unit):\n    return Action.move(Direction.East)\n"
    )
    assert "<think>" not in clean.lower()
    assert "def robot(state, unit):" in clean

    malformed = rrp._strip_or_reject_think_blocks(
        "<think>unterminated\ndef robot(state, unit):\n    return Action.move(Direction.East)\n"
    )
    assert malformed == ""


def test_code_from_completion_prefers_latest_assistant_turn() -> None:
    completion = [
        {
            "role": "user",
            "content": (
                "Current robot.py:\n"
                "```python\n"
                "def robot(state, unit):\n"
                "    return NotAction.move(Direction.East)\n"
                "```"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "```python\n"
                "def robot(state, unit):\n"
                "    return Action.move(Direction.East)\n"
                "```"
            ),
        },
        {"role": "user", "content": "Next round result..."},
    ]
    code = rrp._code_from_completion(completion)
    assert "NotAction" not in code
    assert code == "def robot(state, unit):\n    return Action.move(Direction.East)"


def test_code_from_completion_handles_object_messages() -> None:
    class _Msg:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    completion = [
        _Msg("user", "Ignore this prompt text."),
        _Msg(
            "assistant",
            "def robot(state, unit):\n"
            "    return Action.attack(Direction.North)\n",
        ),
    ]
    code = rrp._code_from_completion(completion)
    assert code == "def robot(state, unit):\n    return Action.attack(Direction.North)"


def test_extract_python_uses_last_fenced_block() -> None:
    text = (
        "```python\n"
        "def robot(state, unit):\n"
        "    return Action.move(Direction.West)\n"
        "```\n"
        "...\n"
        "```python\n"
        "def robot(state, unit):\n"
        "    return Action.attack(Direction.East)\n"
        "```"
    )
    assert rrp._extract_python(text) == (
        "def robot(state, unit):\n"
        "    return Action.attack(Direction.East)"
    )


def test_stratified_sampling_is_seeded_and_middle_count_controlled() -> None:
    def _run_default():
        tasks_a = rrp._build_ladder_vs_humans_tasks()
        tasks_b = rrp._build_ladder_vs_humans_tasks()
        opps_a = tasks_a[0]["opponents"]
        opps_b = tasks_b[0]["opponents"]
        assert opps_a == opps_b
        assert len(opps_a) == 7  # 2 bottom + 3 middle + 2 top

    _with_env(
        {
            "ROBOTRUMBLE_STRATIFIED_OPPONENT_SEED": "2026",
            "ROBOTRUMBLE_STRATIFIED_MIDDLE_COUNT": "3",
        },
        _run_default,
    )

    def _run_middle4():
        tasks = rrp._build_ladder_vs_humans_tasks()
        opps = tasks[0]["opponents"]
        assert len(opps) == 8  # 2 bottom + 4 middle + 2 top

    _with_env(
        {
            "ROBOTRUMBLE_STRATIFIED_OPPONENT_SEED": "2026",
            "ROBOTRUMBLE_STRATIFIED_MIDDLE_COUNT": "4",
        },
        _run_middle4,
    )


def test_env_args_records_stratified_metadata() -> None:
    def _run():
        env = rrp.load_environment(split="ladder_vs_humans_eval", max_examples=2, seed=1337)
        args = env.env_args
        assert args["stratified_middle_count"] == 4
        assert args["stratified_opponent_seed"] == 999
        expected = rrp._select_stratified_opponents(middle_count=4, opponent_seed=999)
        assert args["stratified_opponents"] == expected

    _with_env(
        {
            "ROBOTRUMBLE_STRATIFIED_OPPONENT_SEED": "999",
            "ROBOTRUMBLE_STRATIFIED_MIDDLE_COUNT": "4",
        },
        _run,
    )


def test_load_environment_accepts_replayed_stratified_env_args() -> None:
    selected = rrp._select_stratified_opponents(middle_count=4, opponent_seed=999)
    env = rrp.load_environment(
        split="ladder_vs_humans_eval",
        max_examples=1,
        seed=1337,
        stratified_middle_count=4,
        stratified_opponent_seed=999,
        stratified_opponents=selected,
    )
    assert env.env_args["stratified_middle_count"] == 4
    assert env.env_args["stratified_opponent_seed"] == 999
    assert env.env_args["stratified_opponents"] == selected
    assert env.dataset[0]["opponents"] == selected


def test_canonical_load_environment_accepts_replayed_max_turns() -> None:
    env = rrpc.load_environment(split="eval", max_examples=1, seed=1337, max_turns=35)
    assert env.env_args["max_turns"] == 35


def test_canonical_dataset_scales_to_requested_examples() -> None:
    env = rrpc.load_environment(split="eval", max_examples=128, seed=1337, base_seed=42)
    assert len(env.dataset) == 128
    assert len(env.eval_dataset) == 128
    assert env.env_args["dataset_size"] == 128
    assert env.env_args["eval_dataset_size"] == 128
    offsets = {int(row["info"]["seed_offset"]) for row in env.dataset}
    assert len(offsets) == 128


def test_canonical_default_dataset_sizes_are_not_tiny() -> None:
    env = rrpc.load_environment(split="train", max_examples=-1, seed=9, base_seed=77)
    assert len(env.dataset) == rrpc._DEFAULT_TRAIN_DATASET_SIZE
    assert len(env.eval_dataset) == rrpc._DEFAULT_EVAL_DATASET_SIZE
    assert env.env_args["dataset_size"] == rrpc._DEFAULT_TRAIN_DATASET_SIZE
    assert env.env_args["eval_dataset_size"] == rrpc._DEFAULT_EVAL_DATASET_SIZE


def test_canonical_setup_state_uses_config_base_seed_plus_offset(monkeypatch) -> None:
    monkeypatch.setattr(rrpc, "runner_status", lambda: (True, None))
    monkeypatch.setattr(
        rrpc,
        "get_opponent_source",
        lambda _branch: (True, "def robot(state, unit):\n    return Action.move(Direction.East)\n", None),
    )

    env = rrpc.load_environment(split="eval", max_examples=1, seed=5, base_seed=42, rounds_per_opponent=3)
    seed_offset = int(env.dataset[0]["info"]["seed_offset"])

    # Legacy info["base_seed"] should not override the configured env base_seed.
    state = asyncio.run(env.setup_state({"info": {"base_seed": 999, "seed_offset": seed_offset}}))
    assert state["base_seed"] == 42 + seed_offset


def test_variant_dispatch_loads_canonical_env() -> None:
    env = rrp.load_environment(
        split="eval",
        max_examples=1,
        seed=1337,
        variant="canonical",
        rounds_per_opponent=3,
        turns_per_match=60,
        base_seed=42,
    )
    assert env.env_id == "robotrumble-prime-canonical"
    assert env.env_args["variant"] == "canonical"
    assert env.env_args["rounds_per_opponent"] == 3
    assert env.env_args["turns_per_match"] == 60
    assert env.env_args["base_seed"] == 42
    assert "iteratively editing" in env.system_prompt.lower()


def test_canonical_load_environment_accepts_floor_break_train_profile() -> None:
    env = rrpc.load_environment(
        split="train",
        max_examples=1,
        seed=1337,
        base_seed=42,
        train_profile="contract_recovery_floor_break",
    )
    assert env.env_args["train_profile"] == "contract_recovery_floor_break"


def test_canonical_late_round_quality_tracks_current_opponent_tail() -> None:
    state = {
        "opponents": ["human/a", "human/b"],
        "current_opponent_index": 0,
        "round_history": [
            {"opponent": "human/a", "outcome": "loss"},
            {"opponent": "human/b", "outcome": "win"},
            {"opponent": "human/a", "outcome": "tie"},
            {"opponent": "human/a", "outcome": "win"},
        ],
    }
    # Current-opponent tail is [tie, win] => (0.5 + 1.0) / 2 = 0.75
    assert rrpc.canonical_late_round_quality(state=state) == 0.75


def test_canonical_contract_success_rate_is_inverse_of_failure_rate() -> None:
    state = {
        "round_history": [{"outcome": "win"} for _ in range(4)],
        "contract_failures": 1,
    }
    assert rrpc.canonical_contract_failure_rate(state=state) == 0.25
    assert rrpc.canonical_contract_success_rate(state=state) == 0.75


def test_canonical_reason_rate_metrics() -> None:
    state = {
        "round_history": [
            {"reason": "empty_response"},
            {"reason": "syntax_error"},
            {"reason": "unsafe_constructs"},
            {"reason": "match"},
        ],
        "reason_counts": {
            "empty_response": 1,
            "syntax_error": 1,
            "contract_error": 0,
            "unsafe_constructs": 1,
            "match_infra_error": 0,
            "match": 1,
        },
    }
    assert rrpc.canonical_empty_response_rate(state=state) == 0.25
    assert rrpc.canonical_syntax_error_rate(state=state) == 0.25
    assert rrpc.canonical_contract_error_rate(state=state) == 0.0
    assert rrpc.canonical_unsafe_constructs_rate(state=state) == 0.25
    assert rrpc.canonical_match_infra_error_rate(state=state) == 0.0


def test_canonical_turn_prompt_uses_adaptation_focus_and_robot_tags() -> None:
    state = {
        "opponents": ["human/a", "human/b"],
        "current_opponent_index": 0,
        "rounds_per_opponent": 5,
        "rounds_played_current": 1,
        "wins_current": 0,
        "losses_current": 1,
        "ties_current": 0,
        "defeated_count": 0,
        "current_code": "def robot(state, unit):\n    return Action.move(Direction.East)\n",
        "round_history": [],
    }
    prompt = rrpc._build_turn_prompt(
        state,
        last_round={"outcome": "loss", "reason": "match", "seed": "42-0-0"},
    )
    assert "Adaptation focus:" in prompt
    assert "targeted tactical change" in prompt
    assert "<robot_py>" in prompt
    assert "</robot_py>" in prompt
    assert "Direction input must be API-valid" in prompt
    assert "```python" not in prompt


def test_canonical_competitive_score_counts_ties_as_partial_progress() -> None:
    state = {
        "opponents": ["human/a", "human/b"],
        "defeated_count": 0,
        "rounds_per_opponent": 5,
        "wins_current": 1,
        "ties_current": 2,
        "partial_progress_tieaware": 0.0,
    }
    # (1 + 0.5*2)/5 = 0.4, total opponents=2 => 0.2
    assert rrpc.canonical_competitive_score(state=state) == 0.2


def test_canonical_rubric_weights_train_vs_eval() -> None:
    train_env = rrpc.load_environment(split="train", max_examples=1, seed=7, rounds_per_opponent=3, turns_per_match=40)
    eval_env = rrpc.load_environment(split="eval", max_examples=1, seed=7, rounds_per_opponent=3, turns_per_match=40)

    train_weights = train_env.rubric.rubrics[0].weights
    eval_weights = eval_env.rubric.rubrics[0].weights

    assert train_weights == [0.10, 0.30, 0.30, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert eval_weights == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_canonical_rubric_weights_contract_recovery_profile() -> None:
    env = rrpc.load_environment(
        split="train",
        max_examples=1,
        seed=11,
        rounds_per_opponent=3,
        turns_per_match=40,
        train_profile="contract_recovery",
    )
    weights = env.rubric.rubrics[0].weights
    assert weights == [0.10, 0.28, 0.28, 0.18, 0.10, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert env.env_args["train_profile"] == "contract_recovery"
