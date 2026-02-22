from __future__ import annotations

from typing import Any, Sequence

import verifiers as vf
from datasets import Dataset
from ladder_runner import DEFAULT_FIXED_OPPONENTS, get_opponent_source, run_single_match, runner_status
from robotrumble_prime import _code_from_completion, _contract_gate, _syntax_valid_code, safe_constructs

ENV_ID = "robotrumble-prime-canonical"

SYSTEM_PROMPT = (
    "You are iteratively editing RobotRumble `robot.py` in a long-horizon ladder tournament. "
    "Each turn you must return a full replacement `robot.py` file as plain Python source. "
    "Primary objective: make targeted round-to-round improvements against the current opponent while "
    "preserving previously working logic. "
    "When a round fails, patch one concrete weakness; do not rewrite everything. "
    "No prose. No markdown fences. No <think> blocks."
)

_DEFAULT_CANONICAL_OPPONENTS: tuple[str, ...] = (
    "human/underscore/bot1",
    "human/aaa/jippty5",
    "human/kalkin/maxad",
    "human/mario31313/alpha_13",
    "human/navster8/bash-brothers",
    "human/entropicdrifter/we-are-borg",
    "human/entropicdrifter/seven-of-nine",
    "human/entropicdrifter/gigachad",
)
_DEFAULT_ROUNDS_PER_OPPONENT = 5
_DEFAULT_TURNS_PER_MATCH = 100
_DEFAULT_BASE_SEED = 1337
_DEFAULT_TRAIN_PROFILE = "default"
_DEFAULT_TRAIN_RUBRIC_WEIGHTS = [0.10, 0.30, 0.30, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_CONTRACT_RECOVERY_TRAIN_RUBRIC_WEIGHTS = [0.10, 0.28, 0.28, 0.18, 0.10, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_CONTRACT_RECOVERY_FLOOR_BREAK_TRAIN_RUBRIC_WEIGHTS = [0.10, 0.30, 0.30, 0.18, 0.10, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_DEFAULT_TRAIN_DATASET_SIZE = 256
_DEFAULT_EVAL_DATASET_SIZE = 128
_ROUND_REASON_KEYS: tuple[str, ...] = (
    "match",
    "empty_response",
    "syntax_error",
    "contract_error",
    "unsafe_constructs",
    "match_infra_error",
)

_TASK_PROMPTS: tuple[str, ...] = (
    "Iteratively improve robot.py across the ladder while carrying code forward.",
    "Keep edits compact and robust while climbing ladder opponents.",
    "Prioritize legal actions and stable behavior under long-horizon carryover.",
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _outcome_score(outcome: Any) -> float:
    value = str(outcome)
    if value == "win":
        return 1.0
    if value == "tie":
        return 0.5
    return 0.0


def _current_opponent_rounds(state: dict[str, Any]) -> list[dict[str, Any]]:
    history = state.get("round_history", [])
    if not isinstance(history, list) or not history:
        return []
    current_opponent = _current_opponent(state)
    return [row for row in history if isinstance(row, dict) and row.get("opponent") == current_opponent]


def _reason_count(state: dict[str, Any], reason: str) -> int:
    counts = state.get("reason_counts")
    if isinstance(counts, dict):
        try:
            return max(0, int(counts.get(reason, 0)))
        except (TypeError, ValueError):
            return 0
    rounds = state.get("round_history", [])
    if not isinstance(rounds, list):
        return 0
    return sum(1 for row in rounds if isinstance(row, dict) and row.get("reason") == reason)


def _reason_rate(state: dict[str, Any], reason: str) -> float:
    rounds = state.get("round_history", [])
    if not isinstance(rounds, list) or len(rounds) <= 0:
        return 0.0
    return _clip01(float(_reason_count(state, reason)) / float(len(rounds)))


def _increment_reason_count(state: dict[str, Any], reason: str) -> None:
    counts = state.get("reason_counts")
    if not isinstance(counts, dict):
        counts = {}
        state["reason_counts"] = counts
    current = counts.get(reason, 0)
    try:
        parsed = int(current)
    except (TypeError, ValueError):
        parsed = 0
    counts[reason] = parsed + 1


def canonical_ladder_advancement(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    total = len(state.get("opponents", []))
    if total <= 0:
        return 0.0
    return _clip01(float(state.get("defeated_count", 0)) / float(total))


def canonical_ladder_strength(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0

    total = len(state.get("opponents", []))
    if total <= 0:
        return 0.0

    defeated = float(state.get("defeated_count", 0))
    partial = float(state.get("partial_progress", 0.0))
    if partial <= 0.0 and state.get("stop_reason") is None:
        rounds = int(state.get("rounds_per_opponent", 0))
        wins = int(state.get("wins_current", 0))
        if rounds > 0:
            partial = wins / float(rounds)
    return _clip01((defeated + partial) / float(total))


def canonical_competitive_score(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    total = len(state.get("opponents", []))
    if total <= 0:
        return 0.0

    defeated = float(state.get("defeated_count", 0))
    rounds = int(state.get("rounds_per_opponent", 0))
    if rounds <= 0:
        return _clip01(defeated / float(total))

    # Dense in-opponent signal: ties contribute partial progress.
    wins = int(state.get("wins_current", 0))
    ties = int(state.get("ties_current", 0))
    partial_live = (wins + 0.5 * ties) / float(rounds)
    partial_recorded = float(state.get("partial_progress_tieaware", 0.0))
    partial = max(partial_live, partial_recorded)
    return _clip01((defeated + partial) / float(total))


def canonical_round_win_rate(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    rounds = state.get("round_history", [])
    if not rounds:
        return 0.0
    wins = sum(1 for row in rounds if row.get("outcome") == "win")
    return _clip01(float(wins) / float(len(rounds)))


def canonical_late_round_quality(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    rounds = _current_opponent_rounds(state)
    if not rounds:
        return 0.0
    tail = rounds[-2:] if len(rounds) >= 2 else rounds
    score = sum(_outcome_score(row.get("outcome")) for row in tail) / float(len(tail))
    return _clip01(score)


def canonical_contract_failure_rate(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    rounds_played = len(state.get("round_history", []))
    if rounds_played <= 0:
        return 0.0
    failures = int(state.get("contract_failures", 0))
    return _clip01(float(failures) / float(rounds_played))


def canonical_contract_success_rate(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    return _clip01(1.0 - canonical_contract_failure_rate(state=state))


def canonical_empty_response_rate(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    return _reason_rate(state, "empty_response")


def canonical_syntax_error_rate(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    return _reason_rate(state, "syntax_error")


def canonical_contract_error_rate(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    return _reason_rate(state, "contract_error")


def canonical_unsafe_constructs_rate(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    return _reason_rate(state, "unsafe_constructs")


def canonical_match_infra_error_rate(completion: Any = None, *, state: Any | None = None, **kwargs) -> float:
    if not isinstance(state, dict):
        return 0.0
    return _reason_rate(state, "match_infra_error")


def _replace_prompt_with_user(prompt: Any, user_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(prompt, list):
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            content = msg.get("content")
            if role in {"system", "developer"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


def _current_opponent(state: dict[str, Any]) -> str:
    return str(state["opponents"][int(state["current_opponent_index"])])


def _seed_for_round(state: dict[str, Any]) -> str:
    base_seed = int(state["base_seed"])
    opponent_idx = int(state["current_opponent_index"])
    round_idx = int(state["rounds_played_current"])
    return f"{base_seed}-{opponent_idx}-{round_idx}"


def _format_recent_rounds(state: dict[str, Any], limit: int = 6) -> str:
    rows = state.get("round_history", [])
    if not rows:
        return "none"
    tail = rows[-limit:]
    parts = []
    for row in tail:
        parts.append(
            f"- opp={row['opponent']} round={row['round']} outcome={row['outcome']} reason={row['reason']}"
        )
    return "\n".join(parts)


def _adaptation_focus(last_round: dict[str, Any] | None) -> str:
    if not isinstance(last_round, dict):
        return "Establish a legal, stable baseline policy before adding complexity."
    reason = str(last_round.get("reason", ""))
    outcome = str(last_round.get("outcome", ""))
    if reason != "match":
        return "Fix legality first: syntax/signature/API contract/safe constructs before strategy edits."
    if outcome == "loss":
        return "Apply one targeted tactical change for the observed failure mode; keep other logic intact."
    if outcome == "tie":
        return "Add one decisive adjustment to convert ties into wins while preserving safety."
    return "Keep the winning core policy and make only small robustness/consistency improvements."


def _build_turn_prompt(
    state: dict[str, Any],
    *,
    last_round: dict[str, Any] | None = None,
    opponent_result: dict[str, Any] | None = None,
) -> str:
    opponent = _current_opponent(state)
    rounds_per_opponent = int(state["rounds_per_opponent"])
    rounds_played = int(state["rounds_played_current"])
    round_num = rounds_played + 1
    total = len(state["opponents"])
    defeated = int(state["defeated_count"])

    lines = [
        "Canonical CC:Ladder carryover round.",
        f"Current opponent branch: {opponent}",
        f"Round versus this opponent: {round_num}/{rounds_per_opponent}",
        f"Current opponent score: wins={state['wins_current']} losses={state['losses_current']} ties={state['ties_current']}",
        f"Defeated opponents so far: {defeated}/{total}",
    ]

    if last_round is not None:
        lines.append(
            "Last round result: "
            f"outcome={last_round['outcome']} reason={last_round['reason']} seed={last_round['seed']}"
        )
        lines.append(f"Adaptation focus: {_adaptation_focus(last_round)}")

    if opponent_result is not None:
        lines.append(
            "Previous opponent summary: "
            f"advanced={opponent_result['advanced']} wins={opponent_result['wins']} "
            f"losses={opponent_result['losses']} ties={opponent_result['ties']}"
        )

    lines.extend(
        [
            "",
            "Return a full replacement `robot.py` file.",
            "If you output invalid code, that invalid code still carries over.",
            "Do not include markdown fences, prose, or <think> tags.",
            "Make a small, targeted patch from the current file instead of a full rewrite.",
            "",
            "Hard constraints:",
            "- Keep function signature exactly: def robot(state, unit):",
            "- Return Action.move(...) or Action.attack(...).",
            "- Direction input must be API-valid (Direction.<Dir> or unit.coords.direction_to(...)).",
            "",
            "Current robot.py:",
            "<robot_py>",
            state["current_code"],
            "</robot_py>",
            "",
            "Recent round history:",
            _format_recent_rounds(state),
        ]
    )
    return "\n".join(lines)


def _build_final_summary(state: dict[str, Any], *, failed_opponent: dict[str, Any] | None = None) -> str:
    total = len(state["opponents"])
    defeated = int(state["defeated_count"])
    strength = canonical_ladder_strength(state=state)
    stop_reason = str(state.get("stop_reason", "unknown"))

    lines = [
        "Canonical ladder rollout complete.",
        f"stop_reason={stop_reason}",
        f"ladder_advancement={defeated}/{total} ({canonical_ladder_advancement(state=state):.4f})",
        f"ladder_strength={strength:.4f}",
        f"contract_failures={state.get('contract_failures', 0)}",
        f"match_failures={state.get('match_failures', 0)}",
    ]
    counts = state.get("reason_counts")
    if isinstance(counts, dict):
        reason_stats = ", ".join(
            f"{reason}:{int(counts.get(reason, 0))}" for reason in _ROUND_REASON_KEYS if int(counts.get(reason, 0)) > 0
        )
        if reason_stats:
            lines.append(f"reason_counts={reason_stats}")
    if stop_reason == "infra_error":
        infra_error = str(state.get("infra_error", "")).strip()
        if infra_error:
            lines.append(f"infra_error={infra_error}")
    if failed_opponent is not None:
        lines.append(
            "failed_opponent="
            f"{failed_opponent['opponent']} "
            f"(wins={failed_opponent['wins']}, losses={failed_opponent['losses']}, ties={failed_opponent['ties']})"
        )
    lines.extend(["", "Recent round history:", _format_recent_rounds(state, limit=12)])
    return "\n".join(lines)


class RobotRumblePrimeCanonicalEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        *,
        opponents: Sequence[str],
        rounds_per_opponent: int,
        turns_per_match: int,
        base_seed: int,
        initial_branch: str,
        **kwargs,
    ):
        if rounds_per_opponent < 3 or rounds_per_opponent % 2 == 0:
            raise ValueError("rounds_per_opponent must be odd and >= 3")
        if turns_per_match <= 0:
            raise ValueError("turns_per_match must be positive")
        if not opponents:
            raise ValueError("opponents must be non-empty")

        self.opponents = tuple(opponents)
        self.rounds_per_opponent = int(rounds_per_opponent)
        self.turns_per_match = int(turns_per_match)
        self.base_seed = int(base_seed)
        self.initial_branch = str(initial_branch)

        max_turns = kwargs.pop("max_turns", len(self.opponents) * self.rounds_per_opponent)
        super().__init__(max_turns=max_turns, **kwargs)

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        available, error = runner_status()
        if not available:
            state["error"] = vf.InfraError(f"ladder runner unavailable: {error}")
            return state

        # Keep configured base_seed authoritative; allow only additive per-example
        # offsets for deterministic task diversity.
        state["base_seed"] = self.base_seed
        info = state.get("info")
        if isinstance(info, dict) and "seed_offset" in info:
            try:
                state["base_seed"] = self.base_seed + int(info["seed_offset"])
            except (TypeError, ValueError):
                state["base_seed"] = self.base_seed

        ok, initial_code, source_error = get_opponent_source(self.initial_branch)
        if not ok or not initial_code:
            state["error"] = vf.InfraError(
                f"failed to load initial branch '{self.initial_branch}': {source_error}"
            )
            return state

        state["opponents"] = list(self.opponents)
        state["rounds_per_opponent"] = self.rounds_per_opponent
        state["turns_per_match"] = self.turns_per_match
        state["initial_branch"] = self.initial_branch
        state["current_code"] = initial_code
        state["current_opponent_index"] = 0
        state["rounds_played_current"] = 0
        state["wins_current"] = 0
        state["losses_current"] = 0
        state["ties_current"] = 0
        state["defeated_count"] = 0
        state["partial_progress"] = 0.0
        state["partial_progress_tieaware"] = 0.0
        state["contract_failures"] = 0
        state["match_failures"] = 0
        state["round_history"] = []
        state["opponent_history"] = []
        state["stop_reason"] = None
        state["infra_error"] = None
        state["reason_counts"] = {reason: 0 for reason in _ROUND_REASON_KEYS}

        first_turn_prompt = _build_turn_prompt(state)
        state["prompt"] = _replace_prompt_with_user(state.get("prompt"), first_turn_prompt)
        return state

    async def env_response(self, messages: Any, state: dict[str, Any], **kwargs) -> list[dict[str, str]]:
        completion = state["trajectory"][-1]["completion"]
        candidate_code = _code_from_completion(completion)
        state["current_code"] = candidate_code

        opponent = _current_opponent(state)
        seed = _seed_for_round(state)

        is_empty = not candidate_code
        is_syntax_valid = _syntax_valid_code(candidate_code) if not is_empty else False
        passes_contract = _contract_gate(candidate_code) if is_syntax_valid else False
        is_safe = safe_constructs(completion) > 0.0 if passes_contract else False

        outcome = "loss"
        reason = "contract_error"
        if is_empty:
            reason = "empty_response"
            state["contract_failures"] = int(state["contract_failures"]) + 1
        elif not is_syntax_valid:
            reason = "syntax_error"
            state["contract_failures"] = int(state["contract_failures"]) + 1
        elif not passes_contract:
            reason = "contract_error"
            state["contract_failures"] = int(state["contract_failures"]) + 1
        elif not is_safe:
            reason = "unsafe_constructs"
            state["contract_failures"] = int(state["contract_failures"]) + 1
        else:
            match = run_single_match(
                candidate_code,
                opponent=opponent,
                turns_per_match=int(state["turns_per_match"]),
                seed=seed,
            )
            if not match.available:
                state["match_failures"] = int(state["match_failures"]) + 1
                state["stop_reason"] = "infra_error"
                state["infra_error"] = match.error or "failed to run match"
                reason = "match_infra_error"
            else:
                outcome = match.outcome
                reason = "match"

        if outcome == "win":
            state["wins_current"] = int(state["wins_current"]) + 1
        elif outcome == "loss":
            state["losses_current"] = int(state["losses_current"]) + 1
        else:
            state["ties_current"] = int(state["ties_current"]) + 1

        state["rounds_played_current"] = int(state["rounds_played_current"]) + 1
        rounds_played = int(state["rounds_played_current"])

        round_record = {
            "opponent": opponent,
            "round": rounds_played,
            "outcome": outcome,
            "reason": reason,
            "seed": seed,
        }
        state["round_history"].append(round_record)
        _increment_reason_count(state, reason)

        if state.get("stop_reason") == "infra_error":
            summary = _build_final_summary(state)
            state["final_env_response"] = [{"role": "user", "content": summary}]
            state["error"] = vf.InfraError(str(state.get("infra_error", "failed to run match")))
            return state["final_env_response"]

        rounds_per_opponent = int(state["rounds_per_opponent"])
        if rounds_played < rounds_per_opponent:
            prompt = _build_turn_prompt(state, last_round=round_record)
            return [{"role": "user", "content": prompt}]

        wins = int(state["wins_current"])
        losses = int(state["losses_current"])
        ties = int(state["ties_current"])
        advanced = wins >= (rounds_per_opponent // 2 + 1) and outcome == "win"
        opponent_result = {
            "opponent": opponent,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "advanced": advanced,
        }
        state["opponent_history"].append(opponent_result)

        if advanced:
            state["defeated_count"] = int(state["defeated_count"]) + 1
            state["current_opponent_index"] = int(state["current_opponent_index"]) + 1

            if int(state["current_opponent_index"]) >= len(state["opponents"]):
                state["stop_reason"] = "cleared_ladder"
                state["partial_progress"] = 0.0
                state["partial_progress_tieaware"] = 0.0
                summary = _build_final_summary(state)
                state["final_env_response"] = [{"role": "user", "content": summary}]
                return state["final_env_response"]

            state["rounds_played_current"] = 0
            state["wins_current"] = 0
            state["losses_current"] = 0
            state["ties_current"] = 0
            prompt = _build_turn_prompt(
                state,
                last_round=round_record,
                opponent_result=opponent_result,
            )
            return [{"role": "user", "content": prompt}]

        state["stop_reason"] = "failed_to_advance"
        state["partial_progress"] = wins / float(rounds_per_opponent)
        state["partial_progress_tieaware"] = (wins + 0.5 * ties) / float(rounds_per_opponent)
        summary = _build_final_summary(state, failed_opponent=opponent_result)
        state["final_env_response"] = [{"role": "user", "content": summary}]
        return state["final_env_response"]


def _resolve_dataset_size(split: str, max_examples: int) -> int:
    if max_examples > 0:
        return int(max_examples)
    if split == "train":
        return _DEFAULT_TRAIN_DATASET_SIZE
    return _DEFAULT_EVAL_DATASET_SIZE


def _build_dataset(split: str, max_examples: int, seed: int, *, base_seed: int) -> Dataset:
    if split not in {"train", "eval"}:
        raise ValueError("split must be one of: train, eval")
    count = max(1, _resolve_dataset_size(split, max_examples))
    split_offset = 0 if split == "train" else 1_000_000
    data: list[dict[str, Any]] = []
    for idx in range(count):
        prompt = _TASK_PROMPTS[idx % len(_TASK_PROMPTS)]
        data.append(
            {
                "question": f"{prompt} [task {idx + 1}/{count}]",
                "answer": f"canonical_{split}_seed_{seed}_{idx}",
                "info": {
                    "seed_offset": split_offset + idx,
                    "task_index": idx,
                    "dataset_seed": seed,
                    "base_seed": base_seed,
                },
            }
        )
    ds = Dataset.from_list(data)
    ds = ds.shuffle(seed=seed)
    return ds


def load_environment(
    split: str = "eval",
    max_examples: int = -1,
    seed: int = 42,
    rounds_per_opponent: int = _DEFAULT_ROUNDS_PER_OPPONENT,
    turns_per_match: int = _DEFAULT_TURNS_PER_MATCH,
    base_seed: int = _DEFAULT_BASE_SEED,
    opponents: list[str] | None = None,
    initial_branch: str | None = None,
    train_profile: str = _DEFAULT_TRAIN_PROFILE,
    max_turns: int | None = None,
    **_unused_kwargs: Any,
) -> vf.Environment:
    split = split.lower().strip()
    if split not in {"train", "eval"}:
        raise ValueError("split must be one of: train, eval")

    ladder = tuple(opponents) if opponents else _DEFAULT_CANONICAL_OPPONENTS
    if not ladder:
        ladder = tuple(DEFAULT_FIXED_OPPONENTS)
    initial = initial_branch or ladder[0]

    train_dataset = _build_dataset(split=split, max_examples=max_examples, seed=seed, base_seed=base_seed)
    eval_dataset = _build_dataset(split="eval", max_examples=max_examples, seed=seed + 1, base_seed=base_seed)
    parser = vf.MaybeThinkParser()
    rubric_funcs = [
        canonical_ladder_advancement,
        canonical_ladder_strength,
        canonical_competitive_score,
        canonical_late_round_quality,
        canonical_round_win_rate,
        canonical_contract_success_rate,
        canonical_contract_failure_rate,
        canonical_empty_response_rate,
        canonical_syntax_error_rate,
        canonical_contract_error_rate,
        canonical_unsafe_constructs_rate,
        canonical_match_infra_error_rate,
    ]
    normalized_train_profile = str(train_profile).strip().lower() or _DEFAULT_TRAIN_PROFILE
    if split == "train":
        if normalized_train_profile == "contract_recovery":
            # Preserve gameplay-heavy shaping while nudging toward contract-valid outputs.
            rubric_weights = _CONTRACT_RECOVERY_TRAIN_RUBRIC_WEIGHTS
        elif normalized_train_profile == "contract_recovery_floor_break":
            # Reduce legality floor pressure so training is less likely to get stuck
            # at contract-valid but non-competitive policies.
            rubric_weights = _CONTRACT_RECOVERY_FLOOR_BREAK_TRAIN_RUBRIC_WEIGHTS
        elif normalized_train_profile in {"default", "gameplay"}:
            # Dense shaping for adaptation: prioritize competitive outcomes.
            # Keep legality metrics monitoring-only to avoid a contract-valid floor.
            rubric_weights = _DEFAULT_TRAIN_RUBRIC_WEIGHTS
        else:
            raise ValueError(
                "train_profile must be one of: default, gameplay, contract_recovery, contract_recovery_floor_break"
            )
    else:
        # Eval stays aligned to dense ladder objective for easier promotion gating.
        rubric_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rubric = vf.Rubric(
        funcs=rubric_funcs,
        weights=rubric_weights,
        parser=parser,
    )

    resolved_max_turns = len(ladder) * rounds_per_opponent if max_turns is None else int(max_turns)
    if resolved_max_turns <= 0:
        resolved_max_turns = len(ladder) * rounds_per_opponent

    env_args: dict[str, Any] = {
        "split": split,
        "max_examples": max_examples,
        "seed": seed,
        "rounds_per_opponent": rounds_per_opponent,
        "turns_per_match": turns_per_match,
        "base_seed": base_seed,
        "train_profile": normalized_train_profile,
        "opponents": list(ladder),
        "initial_branch": initial,
        "max_turns": resolved_max_turns,
        "dataset_size": len(train_dataset),
        "eval_dataset_size": len(eval_dataset),
    }

    return RobotRumblePrimeCanonicalEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        parser=parser,
        env_id=ENV_ID,
        env_args=env_args,
        opponents=ladder,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        base_seed=base_seed,
        initial_branch=initial,
        max_turns=resolved_max_turns,
    )
