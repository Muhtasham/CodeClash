import ast
import importlib.util
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

import verifiers as vf
from datasets import Dataset
from ladder_runner import DEFAULT_FIXED_OPPONENTS, evaluate_fixed_ladder, runner_status
from openai.types.chat.chat_completion import ChatCompletion

ENV_ID = "robotrumble-prime"

SYSTEM_PROMPT = (
    "You are writing Python code for RobotRumble. "
    "Return only valid Python source for robot.py. "
    "The code must define exactly `def robot(state, unit):`. "
    "Every return path must return exactly one of "
    "`Action.move(Direction.<North|South|East|West>)` or "
    "`Action.attack(Direction.<North|South|East|West>)`. "
    "Do not import, redefine, or shadow `Action` or `Direction`. "
    "Do not output markdown, prose, or <think> blocks."
)

_PROMPT_API_CONTRACT = (
    "API contract requirements (must follow exactly):\n"
    "- Define exactly: `def robot(state, unit):`\n"
    "- Every `return` inside `robot` must be `Action.move(Direction.X)` or "
    "`Action.attack(Direction.X)`\n"
    "- `X` must be one of `North`, `South`, `East`, `West`\n"
    "- Do not import, redefine, or shadow `Action`/`Direction`\n"
    "- Return only Python code for `robot.py`"
)
_ALLOWED_ACTION_CALLS = {"move", "attack"}
_ALLOWED_DIRECTION_ATTRS = {"North", "South", "East", "West", "NORTH", "SOUTH", "EAST", "WEST"}

TRAIN_TASKS: list[dict[str, str]] = [
    {
        "question": (
            "Write robot.py for RobotRumble with a `robot(state, unit)` function. "
            "Prioritize attacking adjacent enemies; otherwise move toward the nearest enemy."
        ),
        "answer": "robot function with attack-first strategy",
    },
    {
        "question": (
            "Write robot.py where low-health units retreat, healthy units push center, "
            "and all units still attack when adjacent."
        ),
        "answer": "health-aware strategy",
    },
    {
        "question": (
            "Write robot.py that avoids invalid moves, uses Direction safely, and always "
            "returns an Action."
        ),
        "answer": "safe movement with guaranteed Action return",
    },
    {
        "question": (
            "Write robot.py for simple swarm behavior: focus fire nearest enemy, avoid "
            "friendly collisions when possible, attack if adjacent."
        ),
        "answer": "swarm behavior",
    },
    {
        "question": (
            "Write robot.py with deterministic behavior (no randomness) and clear fallback "
            "logic when blocked."
        ),
        "answer": "deterministic fallback strategy",
    },
    {
        "question": (
            "Write robot.py that favors map-center control early and switches to direct chase "
            "when enemies are close."
        ),
        "answer": "center control then chase",
    },
]

EVAL_TASKS: list[dict[str, str]] = [
    {
        "question": (
            "Write robust RobotRumble robot.py with `robot(state, unit)` that handles missing "
            "targets, blocked paths, and always returns an Action."
        ),
        "answer": "robust robot function",
    },
    {
        "question": (
            "Write robot.py focused on syntactic correctness and API correctness using "
            "Action.move/attack and Direction values."
        ),
        "answer": "syntax + API correctness",
    },
    {
        "question": (
            "Write robot.py where adjacent attack is top priority, then nearest-enemy movement, "
            "then safe fallback movement."
        ),
        "answer": "priority-ordered decision policy",
    },
]

_DEFAULT_LADDER_OPPONENTS = list(DEFAULT_FIXED_OPPONENTS)
_BOOTSTRAP_OPPONENTS = ["human/entropicdrifter/seven-of-nine"]
_SEVEN_OF_NINE = "human/entropicdrifter/seven-of-nine"
_WE_ARE_BORG = "human/entropicdrifter/we-are-borg"
_GIGACHAD = "human/entropicdrifter/gigachad"
_STRATIFIED_BOTTOM_POOL: tuple[str, ...] = (
    "human/underscore/bot1",
    "human/aaa/jippty5",
)
_STRATIFIED_MIDDLE_POOL: tuple[str, ...] = (
    "human/kalkin/maxad",
    "human/mario31313/alpha_13",
    "human/lanity/sivuy",
    "human/navster8/bash-brothers",
)
_STRATIFIED_TOP_POOL: tuple[str, ...] = (
    "human/entropicdrifter/we-are-borg",
    "human/entropicdrifter/seven-of-nine",
    "human/entropicdrifter/gigachad",
)
_STRATIFIED_BOTTOM_COUNT = 2
_STRATIFIED_TOP_COUNT = 2
_DEFAULT_STRATIFIED_MIDDLE_COUNT = 3
_ALLOWED_STRATIFIED_MIDDLE_COUNTS = {3, 4}
_DEFAULT_STRATIFIED_OPPONENT_SEED = 2026
_CANONICAL_VARIANTS = {
    "canonical",
    "robotrumble-prime-canonical",
    "robotrumble_prime_canonical",
}

LADDER_TASKS: list[dict[str, Any]] = [
    {
        "question": (
            "Write RobotRumble `robot.py` with deterministic behavior and robust fallback logic. "
            "Prioritize adjacent attacks, then safe pursuit."
        ),
        "answer": "ladder_eval_seed_1337",
        "opponents": _DEFAULT_LADDER_OPPONENTS,
        "rounds_per_opponent": 3,
        "turns_per_match": 100,
        "seed": 1337,
    },
    {
        "question": (
            "Write RobotRumble `robot.py` focused on practical win rate against strong human bots. "
            "Avoid invalid moves and preserve stable behavior."
        ),
        "answer": "ladder_eval_seed_4242",
        "opponents": _DEFAULT_LADDER_OPPONENTS,
        "rounds_per_opponent": 3,
        "turns_per_match": 100,
        "seed": 4242,
    },
    {
        "question": (
            "Write concise and effective RobotRumble policy code. Return only `robot.py` code with "
            "clear attack/move fallback structure."
        ),
        "answer": "ladder_eval_seed_9001",
        "opponents": _DEFAULT_LADDER_OPPONENTS,
        "rounds_per_opponent": 3,
        "turns_per_match": 100,
        "seed": 9001,
    },
    {
        "question": (
            "Write RobotRumble `robot.py` that avoids corner traps, keeps units moving safely, "
            "and still prioritizes adjacent attacks."
        ),
        "answer": "ladder_eval_seed_2026",
        "opponents": [
            "human/entropicdrifter/seven-of-nine",
            "human/entropicdrifter/we-are-borg",
        ],
        "rounds_per_opponent": 3,
        "turns_per_match": 120,
        "seed": 2026,
    },
    {
        "question": (
            "Write RobotRumble `robot.py` with deterministic target selection and no randomness. "
            "Adjacent attack first, then shortest-path pursuit."
        ),
        "answer": "ladder_eval_seed_77",
        "opponents": [
            "human/entropicdrifter/we-are-borg",
            "human/entropicdrifter/gigachad",
            "human/entropicdrifter/seven-of-nine",
        ],
        "rounds_per_opponent": 3,
        "turns_per_match": 100,
        "seed": 77,
    },
    {
        "question": (
            "Write RobotRumble `robot.py` that kites at low health and collapses on nearest enemy "
            "at high health, while always returning an Action."
        ),
        "answer": "ladder_eval_seed_8080",
        "opponents": [
            "human/entropicdrifter/we-are-borg",
            "human/entropicdrifter/gigachad",
        ],
        "rounds_per_opponent": 3,
        "turns_per_match": 120,
        "seed": 8080,
    },
    {
        "question": (
            "Write robust RobotRumble `robot.py` for boss-level play. Keep logic compact, "
            "safe, and aggressive when a clean attack is available."
        ),
        "answer": "ladder_eval_seed_5151",
        "opponents": [
            "human/entropicdrifter/gigachad",
        ],
        "rounds_per_opponent": 5,
        "turns_per_match": 100,
        "seed": 5151,
    },
    {
        "question": (
            "Write short and readable RobotRumble `robot.py` with explicit fallback behavior when "
            "no valid attack target is available."
        ),
        "answer": "ladder_eval_seed_2718",
        "opponents": _DEFAULT_LADDER_OPPONENTS,
        "rounds_per_opponent": 3,
        "turns_per_match": 80,
        "seed": 2718,
    },
    {
        "question": (
            "Write RobotRumble `robot.py` with strong opening pressure, then switch to stable "
            "safe pursuit as the match progresses."
        ),
        "answer": "ladder_eval_seed_1618",
        "opponents": _DEFAULT_LADDER_OPPONENTS,
        "rounds_per_opponent": 5,
        "turns_per_match": 100,
        "seed": 1618,
    },
    {
        "question": (
            "Write RobotRumble `robot.py` that recovers from blocked movement, avoids illegal "
            "actions, and preserves deterministic behavior."
        ),
        "answer": "ladder_eval_seed_31415",
        "opponents": [
            "human/entropicdrifter/seven-of-nine",
            "human/entropicdrifter/we-are-borg",
        ],
        "rounds_per_opponent": 5,
        "turns_per_match": 120,
        "seed": 31415,
    },
]


def _bootstrap_tasks_from_ladder(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bootstrap_tasks: list[dict[str, Any]] = []
    for task in tasks:
        item = dict(task)
        item["opponents"] = list(_BOOTSTRAP_OPPONENTS)
        # Slightly longer matches and odd rounds reduce variance for single-opponent hill-climb.
        item["rounds_per_opponent"] = 5
        item["turns_per_match"] = max(120, int(item.get("turns_per_match", 100)))
        bootstrap_tasks.append(item)
    return bootstrap_tasks


LADDER_BOOTSTRAP_TASKS: list[dict[str, Any]] = _bootstrap_tasks_from_ladder(LADDER_TASKS)

LADDER_VS_HUMANS_TASKS: list[dict[str, Any]] = [
    {
        "question": (
            "Write RobotRumble `robot.py` with deterministic target selection and no randomness. "
            "Adjacent attack first, then shortest-path pursuit."
        ),
        "answer": "ladder_vs_humans_seed_77",
        "seed": 77,
    },
    {
        "question": (
            "Write robust RobotRumble `robot.py` for boss-level play. Keep logic compact, "
            "safe, and aggressive when a clean attack is available."
        ),
        "answer": "ladder_vs_humans_seed_5151",
        "seed": 5151,
    },
    {
        "question": (
            "Write RobotRumble `robot.py` focused on practical win rate against strong human bots. "
            "Avoid invalid moves and preserve stable behavior."
        ),
        "answer": "ladder_vs_humans_seed_4242",
        "seed": 4242,
    },
]


def _stratified_middle_count() -> int:
    raw = os.getenv("ROBOTRUMBLE_STRATIFIED_MIDDLE_COUNT", "").strip()
    if not raw:
        return _DEFAULT_STRATIFIED_MIDDLE_COUNT
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_STRATIFIED_MIDDLE_COUNT
    if value in _ALLOWED_STRATIFIED_MIDDLE_COUNTS:
        return value
    return _DEFAULT_STRATIFIED_MIDDLE_COUNT


def _stratified_opponent_seed() -> int:
    raw = os.getenv("ROBOTRUMBLE_STRATIFIED_OPPONENT_SEED", "").strip()
    if not raw:
        return _DEFAULT_STRATIFIED_OPPONENT_SEED
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_STRATIFIED_OPPONENT_SEED


def _ordered_sample(pool: tuple[str, ...], count: int, rng: random.Random) -> list[str]:
    if count > len(pool):
        raise ValueError(f"Requested {count} opponents from pool of size {len(pool)}")
    picked = set(rng.sample(list(pool), k=count))
    return [branch for branch in pool if branch in picked]


def _select_stratified_opponents(*, middle_count: int, opponent_seed: int) -> list[str]:
    rng = random.Random(opponent_seed)
    return (
        _ordered_sample(_STRATIFIED_BOTTOM_POOL, _STRATIFIED_BOTTOM_COUNT, rng)
        + _ordered_sample(_STRATIFIED_MIDDLE_POOL, middle_count, rng)
        + _ordered_sample(_STRATIFIED_TOP_POOL, _STRATIFIED_TOP_COUNT, rng)
    )


def _coerce_stratified_middle_count(value: Any | None) -> int:
    if value is None:
        return _stratified_middle_count()
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_STRATIFIED_MIDDLE_COUNT
    if parsed in _ALLOWED_STRATIFIED_MIDDLE_COUNTS:
        return parsed
    return _DEFAULT_STRATIFIED_MIDDLE_COUNT


def _coerce_stratified_opponent_seed(value: Any | None) -> int:
    if value is None:
        return _stratified_opponent_seed()
    try:
        return int(value)
    except (TypeError, ValueError):
        return _DEFAULT_STRATIFIED_OPPONENT_SEED


def _coerce_stratified_opponents(value: Any | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        opponents = [str(item).strip() for item in value if str(item).strip()]
        return opponents or None
    return None


def _normalized_variant(value: Any | None) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _load_canonical_env_module():
    try:
        from robotrumble_prime_canonical import load_environment as load_canonical_environment

        return load_canonical_environment
    except ModuleNotFoundError:
        # Fallback for direct file execution/import-by-path in local tests.
        module_path = Path(__file__).with_name("robotrumble_prime_canonical.py")
        spec = importlib.util.spec_from_file_location("robotrumble_prime_canonical", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "load_environment")


def _build_ladder_vs_humans_tasks(
    *,
    middle_count: int | None = None,
    opponent_seed: int | None = None,
    opponents: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected_opponents = _coerce_stratified_opponents(opponents)
    if selected_opponents is None:
        selected_opponents = _select_stratified_opponents(
            middle_count=_coerce_stratified_middle_count(middle_count),
            opponent_seed=_coerce_stratified_opponent_seed(opponent_seed),
        )

    tasks: list[dict[str, Any]] = []
    for task in LADDER_VS_HUMANS_TASKS:
        item = dict(task)
        item["opponents"] = list(selected_opponents)
        item["rounds_per_opponent"] = 3
        item["turns_per_match"] = 100
        tasks.append(item)
    return tasks


class RobotRumblePrimeEnv(vf.SingleTurnEnv):
    async def get_model_response(
        self,
        state: Any,
        prompt: Any,
        client: Any | None = None,
        model: str | None = None,
        oai_tools: Any | None = None,
        sampling_args: Any | None = None,
        message_type: Any | None = None,
    ) -> Any:
        # Prime integration tests run vf-eval with OPENAI_API_KEY as key var.
        # If unset, avoid network auth failure by returning a deterministic
        # synthetic chat completion for smoke-test evaluation only.
        if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("OPENAI_API_KEY"):
            content = (
                "def robot(state, unit):\n"
                "    _ = Direction.NORTH\n"
                "    return Action.wait\n"
            )
            return ChatCompletion.model_validate(
                {
                    "id": "chatcmpl-smoke",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model or "smoke-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )
        return await super().get_model_response(
            state,
            prompt,
            client=client,
            model=model,
            oai_tools=oai_tools,
            sampling_args=sampling_args,
            message_type=message_type,
        )


def _extract_content_text(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if content is None:
        return []
    if hasattr(content, "text"):
        text = getattr(content, "text", None)
        if isinstance(text, str):
            return [text]
    if hasattr(content, "content"):
        nested_content = getattr(content, "content", None)
        nested_parts = _extract_content_text(nested_content)
        if nested_parts:
            return nested_parts
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(chunk, "text"):
                text = getattr(chunk, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(chunk, "content"):
                nested_parts = _extract_content_text(getattr(chunk, "content", None))
                if nested_parts:
                    parts.extend(nested_parts)
            elif isinstance(chunk, str):
                parts.append(chunk)
        return parts
    return []


def _extract_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if hasattr(completion, "content"):
        parts = _extract_content_text(getattr(completion, "content", None))
        if parts:
            return "\n".join(parts).strip()

    if isinstance(completion, list):
        assistant_messages: list[str] = []
        fallback_parts: list[str] = []
        for item in completion:
            role: Any | None = None
            content: Any | None = None
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
            elif hasattr(item, "role") or hasattr(item, "content"):
                role = getattr(item, "role", None)
                content = getattr(item, "content", None)

            parts = _extract_content_text(content)
            if parts:
                joined = "\n".join(parts).strip()
                fallback_parts.append(joined)
                if isinstance(role, str) and role.lower() == "assistant":
                    assistant_messages.append(joined)
            elif content is None:
                fallback_parts.append(str(item))

        # Multi-turn trajectories may include prior user prompts with fenced code.
        # Prefer the latest assistant turn to avoid parsing stale prompt snippets.
        if assistant_messages:
            return assistant_messages[-1].strip()
        return "\n".join(fallback_parts).strip()

    return str(completion).strip()


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)


def _strip_or_reject_think_blocks(text: str) -> str:
    if not text:
        return ""
    sanitized = _THINK_BLOCK_RE.sub("", text)
    # Reject malformed/unclosed think blocks to avoid parsing reasoning text as code.
    if _THINK_OPEN_RE.search(sanitized) or _THINK_CLOSE_RE.search(sanitized):
        return ""
    return sanitized.strip()


def _extract_python(text: str) -> str:
    fenced_python = list(re.finditer(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE))
    if fenced_python:
        return fenced_python[-1].group(1).strip()

    fenced_any = list(re.finditer(r"```\s*(.*?)```", text, flags=re.DOTALL))
    if fenced_any:
        return fenced_any[-1].group(1).strip()

    return text.strip()


def _code_from_completion(completion: Any) -> str:
    text = _extract_text(completion)
    text = _strip_or_reject_think_blocks(text)
    return _extract_python(text)


def _parse_tree(code: str) -> ast.AST | None:
    if not code.strip():
        return None
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _robot_fn_node(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "robot":
            return node
    return None


def _has_value_return(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            return True
    return False


def _target_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
        return names
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
    return names


def _is_direction_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Direction"
        and node.attr in _ALLOWED_DIRECTION_ATTRS
    )


def _is_direction_to_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if len(node.args) != 1 or len(node.keywords) > 0:
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "direction_to"


def _is_direction_expr(node: ast.AST, aliases: dict[str, bool]) -> bool:
    if _is_direction_attr(node) or _is_direction_to_call(node):
        return True
    if isinstance(node, ast.Name):
        return bool(aliases.get(node.id, False))
    if isinstance(node, ast.IfExp):
        return _is_direction_expr(node.body, aliases) and _is_direction_expr(node.orelse, aliases)
    return False


def _collect_direction_aliases(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, bool]:
    aliases: dict[str, bool] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            is_dir_value = _is_direction_expr(node.value, aliases)
            for target in node.targets:
                for name in _target_names(target):
                    aliases[name] = aliases.get(name, True) and is_dir_value
        elif isinstance(node, ast.AnnAssign):
            is_dir_value = _is_direction_expr(node.value, aliases) if node.value is not None else False
            for name in _target_names(node.target):
                aliases[name] = aliases.get(name, True) and is_dir_value
        elif isinstance(node, ast.AugAssign):
            for name in _target_names(node.target):
                aliases[name] = False
    return aliases


def _is_valid_direction_expr(node: ast.AST, aliases: dict[str, bool]) -> bool:
    return _is_direction_expr(node, aliases)


def _is_valid_action_call(node: ast.AST, aliases: dict[str, bool]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "Action"
        and func.attr in _ALLOWED_ACTION_CALLS
    ):
        return False
    if len(node.args) != 1 or len(node.keywords) > 0:
        return False
    return _is_valid_direction_expr(node.args[0], aliases)


def _redefines_or_shadows_api_symbols(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"Action", "Direction"}
        ):
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.rsplit(".", 1)[-1]
                if bound in {"Action", "Direction"}:
                    return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound in {"Action", "Direction"}:
                    return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if {"Action", "Direction"} & _target_names(target):
                    return True
        if isinstance(node, ast.AnnAssign) and {"Action", "Direction"} & _target_names(node.target):
            return True
        if isinstance(node, ast.AugAssign) and {"Action", "Direction"} & _target_names(node.target):
            return True
    return False


def _returns_exact_action_contract(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    aliases = _collect_direction_aliases(fn)
    saw_return = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            return False
        saw_return = True
        if not _is_valid_action_call(node.value, aliases):
            return False
    return saw_return


def _syntax_valid_code(code: str) -> bool:
    return _parse_tree(code) is not None


def _has_robot_signature(code: str) -> bool:
    tree = _parse_tree(code)
    if tree is None:
        return False
    fn = _robot_fn_node(tree)
    if fn is None:
        return False
    all_args = list(fn.args.posonlyargs) + list(fn.args.args)
    if len(all_args) != 2:
        return False
    if fn.args.vararg is not None or fn.args.kwarg is not None or len(fn.args.kwonlyargs) > 0:
        return False
    return all_args[0].arg == "state" and all_args[1].arg == "unit"


def _returns_action(code: str) -> float:
    tree = _parse_tree(code)
    if tree is None:
        return 0.0
    fn = _robot_fn_node(tree)
    if fn is None:
        return 0.0
    if _redefines_or_shadows_api_symbols(tree):
        return 0.0
    return 1.0 if _returns_exact_action_contract(fn) else 0.0


def _contract_gate(code: str) -> bool:
    tree = _parse_tree(code)
    if tree is None:
        return False
    fn = _robot_fn_node(tree)
    if fn is None:
        return False
    if _redefines_or_shadows_api_symbols(tree):
        return False
    return _has_robot_signature(code) and _returns_exact_action_contract(fn)


def has_robot_signature(completion: Any, **kwargs) -> float:
    code = _code_from_completion(completion)
    return 1.0 if _has_robot_signature(code) else 0.0


def returns_action(completion: Any, **kwargs) -> float:
    code = _code_from_completion(completion)
    if not _has_robot_signature(code):
        return 0.0
    return _returns_action(code)


def uses_robotrumble_api(completion: Any, **kwargs) -> float:
    code = _code_from_completion(completion)
    if not code:
        return 0.0
    tree = _parse_tree(code)
    if tree is None:
        return 0.0
    fn = _robot_fn_node(tree)
    if fn is None:
        return 0.0
    if _redefines_or_shadows_api_symbols(tree):
        return 0.0
    return 1.0 if _returns_exact_action_contract(fn) else 0.0


def syntax_valid(completion: Any, **kwargs) -> float:
    code = _code_from_completion(completion)
    return 1.0 if _syntax_valid_code(code) else 0.0


def safe_constructs(completion: Any, **kwargs) -> float:
    code = _code_from_completion(completion)
    if not code:
        return 0.0
    banned = ["os.system(", "subprocess", "eval(", "exec("]
    return 0.0 if any(token in code for token in banned) else 1.0


def ladder_runner_ready(completion: Any, **kwargs) -> float:
    code = _code_from_completion(completion)
    if not _has_robot_signature(code):
        return 0.0
    if not _syntax_valid_code(code):
        return 0.0
    available, _error = runner_status()
    return 1.0 if available else 0.0


def ladder_strength(
    completion: Any,
    *,
    opponents: list[str] | None = None,
    rounds_per_opponent: int = 3,
    turns_per_match: int = 100,
    seed: int = 1337,
    **kwargs,
) -> float:
    code = _code_from_completion(completion)
    if not code:
        return 0.0

    # Hard gate for expensive gameplay scoring.
    if not _contract_gate(code):
        return 0.0
    if not _syntax_valid_code(code):
        return 0.0
    if safe_constructs(completion) <= 0.0:
        return 0.0

    result = evaluate_fixed_ladder(
        code,
        opponents=tuple(opponents) if opponents else DEFAULT_FIXED_OPPONENTS,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        base_seed=seed,
    )
    if not result.available:
        return 0.0
    return result.score


def ladder_advancement(
    completion: Any,
    *,
    opponents: list[str] | None = None,
    rounds_per_opponent: int = 3,
    turns_per_match: int = 100,
    seed: int = 1337,
    **kwargs,
) -> float:
    code = _code_from_completion(completion)
    if not code:
        return 0.0

    # Strict gate before gameplay scoring.
    if not _contract_gate(code):
        return 0.0
    if not _syntax_valid_code(code):
        return 0.0
    if safe_constructs(completion) <= 0.0:
        return 0.0

    opps = tuple(opponents) if opponents else DEFAULT_FIXED_OPPONENTS
    result = evaluate_fixed_ladder(
        code,
        opponents=opps,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        base_seed=seed,
    )
    if not result.available:
        return 0.0
    return result.defeated_count / float(len(opps))


def _single_opponent_eval(
    completion: Any,
    *,
    opponent: str,
    rounds_per_opponent: int = 5,
    turns_per_match: int = 100,
    seed: int = 1337,
) -> tuple[float, float]:
    code = _code_from_completion(completion)
    if not code:
        return 0.0, 0.0

    if not _contract_gate(code):
        return 0.0, 0.0
    if not _syntax_valid_code(code):
        return 0.0, 0.0
    if safe_constructs(completion) <= 0.0:
        return 0.0, 0.0

    result = evaluate_fixed_ladder(
        code,
        opponents=(opponent,),
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        base_seed=seed,
    )
    if not result.available or not result.opponents_evaluated:
        return 0.0, 0.0

    opp_eval = result.opponents_evaluated[0]
    total = opp_eval.wins + opp_eval.losses + opp_eval.ties
    if total <= 0:
        return 0.0, 0.0

    win_rate = opp_eval.wins / float(total)
    advanced = 1.0 if opp_eval.advanced else 0.0
    return win_rate, advanced


def winrate_vs_seven_of_nine(
    completion: Any,
    *,
    rounds_per_opponent: int = 5,
    turns_per_match: int = 100,
    seed: int = 1337,
    **kwargs,
) -> float:
    win_rate, _advanced = _single_opponent_eval(
        completion,
        opponent=_SEVEN_OF_NINE,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        seed=seed,
    )
    return win_rate


def winrate_vs_we_are_borg(
    completion: Any,
    *,
    rounds_per_opponent: int = 5,
    turns_per_match: int = 100,
    seed: int = 1337,
    **kwargs,
) -> float:
    win_rate, _advanced = _single_opponent_eval(
        completion,
        opponent=_WE_ARE_BORG,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        seed=seed,
    )
    return win_rate


def winrate_vs_gigachad(
    completion: Any,
    *,
    rounds_per_opponent: int = 5,
    turns_per_match: int = 100,
    seed: int = 1337,
    **kwargs,
) -> float:
    win_rate, _advanced = _single_opponent_eval(
        completion,
        opponent=_GIGACHAD,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        seed=seed,
    )
    return win_rate


def advances_vs_seven_of_nine(
    completion: Any,
    *,
    rounds_per_opponent: int = 5,
    turns_per_match: int = 100,
    seed: int = 1337,
    **kwargs,
) -> float:
    _win_rate, advanced = _single_opponent_eval(
        completion,
        opponent=_SEVEN_OF_NINE,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        seed=seed,
    )
    return advanced


def advances_vs_we_are_borg(
    completion: Any,
    *,
    rounds_per_opponent: int = 5,
    turns_per_match: int = 100,
    seed: int = 1337,
    **kwargs,
) -> float:
    _win_rate, advanced = _single_opponent_eval(
        completion,
        opponent=_WE_ARE_BORG,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        seed=seed,
    )
    return advanced


def advances_vs_gigachad(
    completion: Any,
    *,
    rounds_per_opponent: int = 5,
    turns_per_match: int = 100,
    seed: int = 1337,
    **kwargs,
) -> float:
    _win_rate, advanced = _single_opponent_eval(
        completion,
        opponent=_GIGACHAD,
        rounds_per_opponent=rounds_per_opponent,
        turns_per_match=turns_per_match,
        seed=seed,
    )
    return advanced


def _build_dataset(
    split: str,
    max_examples: int,
    seed: int,
    *,
    stratified_middle_count: int | None = None,
    stratified_opponent_seed: int | None = None,
    stratified_opponents: Sequence[str] | None = None,
) -> Dataset:
    split_to_data: dict[str, list[dict[str, Any]]] = {
        "train": TRAIN_TASKS,
        "eval": EVAL_TASKS,
        "ladder_train": LADDER_TASKS,
        "ladder_eval": LADDER_TASKS,
        "ladder_bootstrap_train": LADDER_BOOTSTRAP_TASKS,
        "ladder_bootstrap_eval": LADDER_BOOTSTRAP_TASKS,
    }
    if split == "ladder_vs_humans_eval":
        data = _build_ladder_vs_humans_tasks(
            middle_count=stratified_middle_count,
            opponent_seed=stratified_opponent_seed,
            opponents=stratified_opponents,
        )
    else:
        data = split_to_data[split]
    contracted: list[dict[str, Any]] = []
    for item in data:
        row = dict(item)
        question = row.get("question")
        if isinstance(question, str) and _PROMPT_API_CONTRACT not in question:
            row["question"] = f"{question.strip()}\n\n{_PROMPT_API_CONTRACT}"
        contracted.append(row)
    ds = Dataset.from_list(contracted)
    ds = ds.shuffle(seed=seed)
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


def load_environment(
    split: str = "train",
    max_examples: int = -1,
    seed: int = 42,
    stratified_middle_count: int | None = None,
    stratified_opponent_seed: int | None = None,
    stratified_opponents: list[str] | None = None,
    variant: str | None = None,
    rounds_per_opponent: int | None = None,
    turns_per_match: int | None = None,
    base_seed: int | None = None,
    opponents: list[str] | None = None,
    initial_branch: str | None = None,
    max_turns: int | None = None,
    **_unused_kwargs: Any,
) -> vf.Environment:
    """
    Local RobotRumble-focused environment for Prime eval/RL bootstrapping.

    Args:
        split: one of "train", "eval", "ladder_train", "ladder_eval",
            "ladder_bootstrap_train", "ladder_bootstrap_eval", "ladder_vs_humans_eval".
        max_examples: Positive value limits dataset size; -1 uses all.
        seed: Shuffle seed.
        variant: "canonical" routes to robotrumble-prime-canonical MultiTurn env.
    """
    split = split.lower().strip()
    normalized_variant = _normalized_variant(variant)
    if normalized_variant and normalized_variant not in _CANONICAL_VARIANTS:
        raise ValueError(
            "variant must be one of: canonical, robotrumble-prime-canonical, "
            "robotrumble_prime_canonical"
        )

    if normalized_variant in _CANONICAL_VARIANTS:
        load_canonical_environment = _load_canonical_env_module()
        canonical_kwargs: dict[str, Any] = {
            "split": split,
            "max_examples": max_examples,
            "seed": seed,
        }
        if rounds_per_opponent is not None:
            canonical_kwargs["rounds_per_opponent"] = rounds_per_opponent
        if turns_per_match is not None:
            canonical_kwargs["turns_per_match"] = turns_per_match
        if base_seed is not None:
            canonical_kwargs["base_seed"] = base_seed
        if opponents is not None:
            canonical_kwargs["opponents"] = opponents
        if initial_branch is not None:
            canonical_kwargs["initial_branch"] = initial_branch
        if max_turns is not None:
            canonical_kwargs["max_turns"] = max_turns
        canonical_kwargs.update(_unused_kwargs)

        env = load_canonical_environment(**canonical_kwargs)
        if isinstance(env.env_args, dict):
            env.env_args["variant"] = "canonical"
        return env

    valid_splits = {
        "train",
        "eval",
        "ladder_train",
        "ladder_eval",
        "ladder_bootstrap_train",
        "ladder_bootstrap_eval",
        "ladder_vs_humans_eval",
    }
    if split not in valid_splits:
        raise ValueError(
            "split must be one of: train, eval, ladder_train, ladder_eval, "
            "ladder_bootstrap_train, ladder_bootstrap_eval, ladder_vs_humans_eval"
        )

    resolved_middle_count = _coerce_stratified_middle_count(stratified_middle_count)
    resolved_opponent_seed = _coerce_stratified_opponent_seed(stratified_opponent_seed)
    resolved_opponents = _coerce_stratified_opponents(stratified_opponents)

    if resolved_opponents is not None and stratified_middle_count is None:
        inferred_middle = len(resolved_opponents) - _STRATIFIED_BOTTOM_COUNT - _STRATIFIED_TOP_COUNT
        if inferred_middle in _ALLOWED_STRATIFIED_MIDDLE_COUNTS:
            resolved_middle_count = inferred_middle

    if resolved_opponents is None:
        resolved_opponents = _select_stratified_opponents(
            middle_count=resolved_middle_count,
            opponent_seed=resolved_opponent_seed,
        )

    train_dataset = _build_dataset(
        split=split,
        max_examples=max_examples,
        seed=seed,
        stratified_middle_count=resolved_middle_count,
        stratified_opponent_seed=resolved_opponent_seed,
        stratified_opponents=resolved_opponents,
    )
    if split.startswith("ladder_bootstrap_"):
        eval_split = "ladder_bootstrap_eval"
    elif split == "ladder_vs_humans_eval":
        eval_split = "ladder_vs_humans_eval"
    elif split.startswith("ladder_"):
        eval_split = "ladder_eval"
    else:
        eval_split = "eval"
    eval_dataset = _build_dataset(
        split=eval_split,
        max_examples=max_examples,
        seed=seed + 1,
        stratified_middle_count=resolved_middle_count,
        stratified_opponent_seed=resolved_opponent_seed,
        stratified_opponents=resolved_opponents,
    )

    parser = vf.MaybeThinkParser()

    if split in {"train", "eval"}:
        rubric = vf.Rubric(
            funcs=[
                has_robot_signature,
                returns_action,
                uses_robotrumble_api,
                syntax_valid,
                safe_constructs,
            ],
            # Shape toward contract + parse correctness before gameplay.
            weights=[0.30, 0.20, 0.15, 0.30, 0.05],
            parser=parser,
        )
    elif split == "ladder_vs_humans_eval":
        rubric = vf.Rubric(
            funcs=[
                ladder_strength,
                ladder_advancement,
            ],
            # Dense stratified reward; strict advancement is tracked for monitoring.
            weights=[1.0, 0.0],
            parser=parser,
        )
    else:
        rubric = vf.Rubric(
            funcs=[
                ladder_advancement,
            ],
            # v0.5: strict CC:Ladder-aligned reward. No shaping, no partial within-opponent credit.
            weights=[1.0],
            parser=parser,
        )

    env_args: dict[str, Any] = {"split": split, "max_examples": max_examples, "seed": seed}
    if split == "ladder_vs_humans_eval":
        env_args.update(
            {
                "stratified_middle_count": resolved_middle_count,
                "stratified_opponent_seed": resolved_opponent_seed,
                "stratified_opponents": list(resolved_opponents),
            }
        )

    return RobotRumblePrimeEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        parser=parser,
        env_id=ENV_ID,
        env_args=env_args,
    )
