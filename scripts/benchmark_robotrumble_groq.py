#!/usr/bin/env python3
"""Benchmark Groq-hosted models on RobotRumble against a fixed human baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from codeclash.constants import RESULT_TIE

DEFAULT_MODEL_IDS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

DISCOVERY_EXCLUDE_SUBSTRINGS = (
    "whisper",
    "guard",
    "prompt-guard",
    "safeguard",
    "compound",
    "tts",
    "playai",
)

GAME_PROMPT_TEMPLATE = """You are a software developer ({{player_id}}) competing in a coding game called RobotRumble.
RobotRumble is a turn-based coding battle where you program a team of robots in Python to move, attack, and outmaneuver your opponent on a grid.
Every decision is driven by your code, and victory comes from crafting logic that positions robots smartly, times attacks well, and adapts over the 100-turn match.
NOTE: Please ensure that your code runs efficiently (under 60 seconds). Code that exceeds this run time will automatically forfeit the round.

The game is played in __ROUNDS__ rounds. For every round, you (and your competitor) edit program code that controls your bot. This is round {{round}}.
After you and your competitor finish editing your codebases, the game is run automatically.

Your task: improve the bot in `robot.py`, located in {{working_dir}}.
{{working_dir}} is your codebase, which contains both your bot and supporting assets."""

CAPACITY_ERROR_PATTERNS = (
    "capacity_exceeded",
    "tier capacity exceeded",
    "client error '498",
)


@dataclass
class RoundScore:
    round_number: int
    model_score: int
    opponent_score: int
    tie_score: int
    winner: str


@dataclass
class BenchmarkResult:
    model_id: str
    litellm_model_name: str
    player_name: str
    config_path: Path
    metadata_path: Path | None
    round_scores: list[RoundScore]
    service_tier_requested: str | None
    service_tier_used: str | None
    run_attempts: int
    error: str | None = None

    @property
    def model_total(self) -> int:
        return sum(score.model_score for score in self.round_scores)

    @property
    def opponent_total(self) -> int:
        return sum(score.opponent_score for score in self.round_scores)


@dataclass(frozen=True)
class ResultValidation:
    valid_result: bool
    status: str
    reason: str | None = None


def yaml_quote(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-")


def normalize_groq_model_id(model_id: str) -> str:
    return model_id.removeprefix("groq/").strip()


def service_tier_label(service_tier: str | None) -> str:
    return service_tier if service_tier else "default"


def build_service_tier_plan(requested_service_tier: str | None) -> list[str | None]:
    plan: list[str | None] = [requested_service_tier]
    if requested_service_tier == "flex":
        for fallback in (None, "on_demand", "auto"):
            if fallback not in plan:
                plan.append(fallback)
    return plan


def parse_model_arg(models_arg: str | None) -> list[str]:
    if not models_arg:
        return list(DEFAULT_MODEL_IDS)
    return [normalize_groq_model_id(x) for x in models_arg.split(",") if x.strip()]


def is_benchmark_candidate(model_id: str) -> bool:
    model_id = model_id.lower()
    return not any(fragment in model_id for fragment in DISCOVERY_EXCLUDE_SUBSTRINGS)


def default_docker_platform() -> str | None:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "linux/amd64"
    return None


def fetch_groq_model_ids(api_key: str) -> list[str]:
    request = Request(
        "https://api.groq.com/openai/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Groq model discovery failed with status {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Groq model discovery failed: {exc.reason}") from exc

    data = payload.get("data", [])
    model_ids = [entry.get("id", "") for entry in data if isinstance(entry, dict) and entry.get("id")]
    return sorted(set(model_ids))


def render_config(
    *,
    rounds: int,
    sims_per_round: int,
    player_name: str,
    litellm_model_name: str,
    temperature: float,
    service_tier: str | None,
    starter_branch: str,
    opponent_name: str,
    opponent_branch: str,
    docker_platform: str | None,
) -> str:
    prompt = GAME_PROMPT_TEMPLATE.replace("__ROUNDS__", str(rounds))
    lines = [
        "tournament:",
        f"  rounds: {rounds}",
        "game:",
        "  name: RobotRumble",
        f"  sims_per_round: {sims_per_round}",
    ]
    if docker_platform:
        lines.append(f"  docker_platform: {docker_platform}")
    lines.extend(
        [
            "  args:",
            "    raw: true",
            "players:",
            "- agent: mini",
            f"  name: {yaml_quote(player_name)}",
            f"  branch_init: {yaml_quote(starter_branch)}",
            "  config:",
            "    agent: !include mini/default.yaml",
            "    model:",
            f"      model_name: {yaml_quote(litellm_model_name)}",
            "      model_kwargs:",
            f"        temperature: {temperature}",
            *( [f"        service_tier: {service_tier}"] if service_tier else [] ),
            "- agent: dummy",
            f"  name: {yaml_quote(opponent_name)}",
            f"  branch_init: {yaml_quote(opponent_branch)}",
            "prompts:",
            "  game_description: |-",
        ]
    )
    lines.extend(f"    {line}" if line else "" for line in prompt.splitlines())
    return "\n".join(lines) + "\n"


def write_config(
    *,
    config_dir: Path,
    model_id: str,
    rounds: int,
    sims_per_round: int,
    player_name: str,
    litellm_model_name: str,
    temperature: float,
    service_tier: str | None,
    starter_branch: str,
    opponent_name: str,
    opponent_branch: str,
    docker_platform: str | None,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    tier_slug = slugify(service_tier_label(service_tier))
    filename = (
        f"RobotRumble__groq_{slugify(model_id)}__vs__{slugify(opponent_name)}"
        f"__r{rounds}__s{sims_per_round}__tier-{tier_slug}.yaml"
    )
    config_path = config_dir / filename
    config_path.write_text(
        render_config(
            rounds=rounds,
            sims_per_round=sims_per_round,
            player_name=player_name,
            litellm_model_name=litellm_model_name,
            temperature=temperature,
            service_tier=service_tier,
            starter_branch=starter_branch,
            opponent_name=opponent_name,
            opponent_branch=opponent_branch,
            docker_platform=docker_platform,
        )
    )
    return config_path


def run_tournament(
    *,
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
    suffix: str,
    keep_containers: bool,
    model_retry_attempts: int,
) -> Path:
    command = [
        sys.executable,
        "main.py",
        str(config_path),
        "-o",
        str(output_dir),
        "-s",
        suffix,
    ]
    if keep_containers:
        command.append("-k")

    env = os.environ.copy()
    env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = str(model_retry_attempts)
    result = subprocess.run(command, cwd=repo_root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Tournament run failed with exit code {result.returncode} for {config_path.name}")

    folders = sorted(output_dir.glob(f"*{suffix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in folders:
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            return metadata_path

    raise FileNotFoundError(f"Could not find metadata.json for run suffix '{suffix}' in {output_dir}")


def has_capacity_error(output_dir: Path, suffix: str) -> bool:
    pattern_set = tuple(pattern.lower() for pattern in CAPACITY_ERROR_PATTERNS)
    folders = sorted(output_dir.glob(f"*{suffix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in folders:
        for log_name in ("tournament.log", "everything.log", "game.log"):
            log_path = folder / log_name
            if not log_path.exists():
                continue
            with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line_l = line.lower()
                    if any(pattern in line_l for pattern in pattern_set):
                        return True
    return False


def parse_round_scores(metadata_path: Path, *, player_name: str, opponent_name: str) -> list[RoundScore]:
    metadata = json.loads(metadata_path.read_text())
    round_stats = metadata.get("round_stats", {})

    scores: list[RoundScore] = []
    for round_key in sorted(round_stats.keys(), key=lambda x: int(x)):
        round_number = int(round_key)
        if round_number == 0:
            continue
        stats = round_stats[round_key]
        score_map = stats.get("scores", {})
        scores.append(
            RoundScore(
                round_number=round_number,
                model_score=int(score_map.get(player_name, 0)),
                opponent_score=int(score_map.get(opponent_name, 0)),
                tie_score=int(score_map.get(RESULT_TIE, 0)),
                winner=stats.get("winner", RESULT_TIE),
            )
        )
    return scores


def classify_result(
    result: BenchmarkResult,
    *,
    expected_rounds: int,
    dry_run: bool,
) -> ResultValidation:
    if dry_run:
        return ResultValidation(
            valid_result=False,
            status="dry_run",
            reason="Run not executed (`--dry-run`).",
        )
    if result.error:
        return ResultValidation(valid_result=False, status="error", reason=result.error)
    if result.metadata_path is None:
        return ResultValidation(
            valid_result=False,
            status="invalid_missing_metadata",
            reason="Missing metadata_path.",
        )
    observed_rounds = len(result.round_scores)
    if observed_rounds != expected_rounds:
        return ResultValidation(
            valid_result=False,
            status="invalid_incomplete_rounds",
            reason=f"Expected {expected_rounds} rounds, found {observed_rounds}.",
        )
    return ResultValidation(valid_result=True, status="valid")


def print_model_discovery(model_ids: list[str]) -> None:
    print(f"Discovered {len(model_ids)} Groq models:")
    for model_id in model_ids:
        print(f"- {model_id}")


def print_summary(
    results: list[BenchmarkResult],
    *,
    summary_path: Path,
    expected_rounds: int,
    dry_run: bool,
) -> None:
    print("\nRobotRumble Groq benchmark summary")
    print("=" * 80)
    for result in results:
        validation = classify_result(result, expected_rounds=expected_rounds, dry_run=dry_run)
        if not validation.valid_result:
            print(
                f"{result.model_id}: {validation.status.upper()} - {validation.reason} "
                f"(requested={service_tier_label(result.service_tier_requested)}, "
                f"attempts={result.run_attempts})"
            )
            continue
        round_str = ", ".join(
            f"R{score.round_number}:{score.model_score}-{score.opponent_score}-{score.tie_score}"
            for score in result.round_scores
        )
        print(
            f"{result.model_id:<32} total={result.model_total:>4} | "
            f"opponent={result.opponent_total:>4} | tier={service_tier_label(result.service_tier_used):<10} | "
            f"attempts={result.run_attempts} | rounds [{round_str}]"
        )
    print("=" * 80)
    print(f"Summary written to {summary_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Benchmark Groq-hosted models on RobotRumble vs a human baseline.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated Groq model IDs. Use bare IDs (e.g., openai/gpt-oss-120b) "
            "or prefixed IDs (e.g., groq/openai/gpt-oss-120b)."
        ),
    )
    parser.add_argument(
        "--discover-models",
        action="store_true",
        help="Fetch active model IDs from Groq `/models` endpoint instead of using the default shortlist.",
    )
    parser.add_argument(
        "--list-models-only",
        action="store_true",
        help="List discovered model IDs and exit. Requires --discover-models.",
    )
    parser.add_argument(
        "--include-non-text-models",
        action="store_true",
        help="When discovering models, keep safety/audio/system models instead of filtering them out.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Optional cap on number of models to benchmark after filtering (0 = no cap).",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Tournament rounds per model (default: 3).")
    parser.add_argument("--sims-per-round", type=int, default=250, help="Simulations per round (default: 250).")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature (default: 0.2).")
    parser.add_argument(
        "--agent-step-limit",
        type=int,
        default=None,
        help="Deprecated. Step/cost overrides are disabled for parity with standard benchmark configs.",
    )
    parser.add_argument(
        "--agent-cost-limit",
        type=float,
        default=None,
        help="Deprecated. Step/cost overrides are disabled for parity with standard benchmark configs.",
    )
    parser.add_argument(
        "--service-tier",
        type=str,
        choices=["flex", "on_demand", "auto"],
        default=None,
        help="Optional Groq service tier for chat requests (e.g. flex).",
    )
    parser.add_argument(
        "--starter-branch",
        type=str,
        default="starter/python",
        help="Branch used to initialize model codebase (default: starter/python).",
    )
    parser.add_argument(
        "--opponent-branch",
        type=str,
        default="human/entropicdrifter/seven-of-nine",
        help="Fixed human opponent branch (default: human/entropicdrifter/seven-of-nine).",
    )
    parser.add_argument(
        "--opponent-name",
        type=str,
        default="seven-of-nine",
        help="Display name of human opponent in config (default: seven-of-nine).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/ablations/vs_human/generated_groq"),
        help="Directory for generated benchmark configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/groq_robotrumble"),
        help="Directory where tournament logs will be written.",
    )
    parser.add_argument(
        "--suffix-prefix",
        type=str,
        default="groq-robotrumble",
        help="Prefix used for run suffixes to find outputs.",
    )
    parser.add_argument(
        "--docker-platform",
        type=str,
        default=None,
        help=(
            "Docker platform passed to game containers (for example linux/amd64). "
            "Default: linux/amd64 on ARM hosts, unset on x86 hosts."
        ),
    )
    parser.add_argument("--keep-containers", action="store_true", help="Pass -k to keep Docker containers.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print planned commands without running tournaments.",
    )
    parser.add_argument(
        "--model-retry-attempts",
        type=int,
        default=8,
        help="Retry attempts for model API errors during each tournament run (default: 8).",
    )
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model_ids = parse_model_arg(args.models)

    if args.list_models_only and not args.discover_models:
        raise SystemExit("--list-models-only requires --discover-models.")

    if args.discover_models:
        if not api_key:
            raise SystemExit("GROQ_API_KEY must be set to discover models from Groq.")
        model_ids = fetch_groq_model_ids(api_key)
        if not args.include_non_text_models:
            model_ids = [model_id for model_id in model_ids if is_benchmark_candidate(model_id)]

    if args.max_models > 0:
        model_ids = model_ids[: args.max_models]

    if not model_ids:
        raise SystemExit("No models selected. Provide --models or use --discover-models.")

    if args.agent_step_limit is not None or args.agent_cost_limit is not None:
        raise SystemExit(
            "Step/cost overrides are disabled in this runner. "
            "Generated configs use `agent: !include mini/default.yaml` for parity."
        )

    if args.list_models_only:
        print_model_discovery(model_ids)
        return

    if not args.dry_run and not api_key:
        raise SystemExit("GROQ_API_KEY must be set to run benchmarks.")

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = (repo_root / args.config_dir).resolve()
    docker_platform = args.docker_platform if args.docker_platform is not None else default_docker_platform()
    timestamp = time.strftime("%y%m%d%H%M%S")

    results: list[BenchmarkResult] = []

    for idx, model_id in enumerate(model_ids, start=1):
        normalized_model_id = normalize_groq_model_id(model_id)
        litellm_model_name = f"groq/{normalized_model_id}"
        player_name = slugify(normalized_model_id)
        service_tier_plan = build_service_tier_plan(args.service_tier)

        if args.dry_run:
            for attempt_idx, service_tier in enumerate(service_tier_plan, start=1):
                config_path = write_config(
                    config_dir=config_dir,
                    model_id=normalized_model_id,
                    rounds=args.rounds,
                    sims_per_round=args.sims_per_round,
                    player_name=player_name,
                    litellm_model_name=litellm_model_name,
                    temperature=args.temperature,
                    service_tier=service_tier,
                    starter_branch=args.starter_branch,
                    opponent_name=args.opponent_name,
                    opponent_branch=args.opponent_branch,
                    docker_platform=docker_platform,
                )
                suffix = (
                    f"{args.suffix_prefix}.{timestamp}.{idx:02d}.{slugify(normalized_model_id)}"
                    f".a{attempt_idx}.{slugify(service_tier_label(service_tier))}"
                )
                print(
                    f"[dry-run] model={litellm_model_name} tier={service_tier_label(service_tier)} "
                    f"config={config_path} command={sys.executable} main.py {config_path} "
                    f"-o {output_dir} -s {suffix}"
                )
            results.append(
                BenchmarkResult(
                    model_id=normalized_model_id,
                    litellm_model_name=litellm_model_name,
                    player_name=player_name,
                    config_path=config_path,
                    metadata_path=None,
                    round_scores=[],
                    service_tier_requested=args.service_tier,
                    service_tier_used=args.service_tier,
                    run_attempts=len(service_tier_plan),
                    error="dry_run_not_executed",
                )
            )
            continue

        config_path: Path | None = None
        metadata_path: Path | None = None
        round_scores: list[RoundScore] = []
        final_error: str | None = None
        used_service_tier: str | None = None
        attempts_made = 0

        for attempt_idx, service_tier in enumerate(service_tier_plan, start=1):
            attempts_made = attempt_idx
            config_path = write_config(
                config_dir=config_dir,
                model_id=normalized_model_id,
                rounds=args.rounds,
                sims_per_round=args.sims_per_round,
                player_name=player_name,
                litellm_model_name=litellm_model_name,
                temperature=args.temperature,
                service_tier=service_tier,
                starter_branch=args.starter_branch,
                opponent_name=args.opponent_name,
                opponent_branch=args.opponent_branch,
                docker_platform=docker_platform,
            )
            suffix = (
                f"{args.suffix_prefix}.{timestamp}.{idx:02d}.{slugify(normalized_model_id)}"
                f".a{attempt_idx}.{slugify(service_tier_label(service_tier))}"
            )
            try:
                metadata_path = run_tournament(
                    repo_root=repo_root,
                    config_path=config_path,
                    output_dir=output_dir,
                    suffix=suffix,
                    keep_containers=args.keep_containers,
                    model_retry_attempts=args.model_retry_attempts,
                )
                round_scores = parse_round_scores(
                    metadata_path, player_name=player_name, opponent_name=args.opponent_name
                )
                used_service_tier = service_tier
                final_error = None
                break
            except Exception as exc:
                final_error = str(exc)
                capacity_error = has_capacity_error(output_dir=output_dir, suffix=suffix)
                has_fallback = attempt_idx < len(service_tier_plan)
                if capacity_error and has_fallback:
                    next_service_tier = service_tier_plan[attempt_idx]
                    print(
                        f"[capacity-fallback] model={normalized_model_id} "
                        f"tier={service_tier_label(service_tier)} failed with capacity error; "
                        f"retrying with tier={service_tier_label(next_service_tier)}"
                    )
                    continue
                break

        results.append(
            BenchmarkResult(
                model_id=normalized_model_id,
                litellm_model_name=litellm_model_name,
                player_name=player_name,
                config_path=config_path if config_path is not None else Path(""),
                metadata_path=metadata_path,
                round_scores=round_scores,
                service_tier_requested=args.service_tier,
                service_tier_used=used_service_tier,
                run_attempts=attempts_made,
                error=final_error,
            )
        )

    result_entries: list[dict[str, object]] = []
    valid_entries: list[dict[str, object]] = []
    for result in results:
        validation = classify_result(result, expected_rounds=args.rounds, dry_run=args.dry_run)
        entry = {
            "model_id": result.model_id,
            "litellm_model_name": result.litellm_model_name,
            "player_name": result.player_name,
            "config_path": str(result.config_path),
            "metadata_path": str(result.metadata_path) if result.metadata_path else None,
            "model_total": result.model_total,
            "opponent_total": result.opponent_total,
            "round_scores": [
                {
                    "round_number": score.round_number,
                    "model_score": score.model_score,
                    "opponent_score": score.opponent_score,
                    "tie_score": score.tie_score,
                    "winner": score.winner,
                }
                for score in result.round_scores
            ],
            "service_tier_requested": result.service_tier_requested,
            "service_tier_used": result.service_tier_used,
            "run_attempts": result.run_attempts,
            "error": result.error,
            "valid_result": validation.valid_result,
            "result_status": validation.status,
            "validation_reason": validation.reason,
        }
        result_entries.append(entry)
        if validation.valid_result:
            valid_entries.append(entry)

    valid_totals = [int(entry["model_total"]) for entry in valid_entries]
    valid_opp_totals = [int(entry["opponent_total"]) for entry in valid_entries]
    leaderboard = sorted(
        (
            {
                "model_id": str(entry["model_id"]),
                "model_total": int(entry["model_total"]),
                "opponent_total": int(entry["opponent_total"]),
                "margin": int(entry["model_total"]) - int(entry["opponent_total"]),
            }
            for entry in valid_entries
        ),
        key=lambda item: (item["margin"], item["model_total"]),
        reverse=True,
    )

    summary = {
        "created_at": int(time.time()),
        "rounds": args.rounds,
        "sims_per_round": args.sims_per_round,
        "agent_step_limit": args.agent_step_limit,
        "agent_cost_limit": args.agent_cost_limit,
        "docker_platform": docker_platform,
        "service_tier": args.service_tier,
        "model_retry_attempts": args.model_retry_attempts,
        "opponent_name": args.opponent_name,
        "opponent_branch": args.opponent_branch,
        "starter_branch": args.starter_branch,
        "aggregation_excludes_invalid_results": True,
        "aggregate": {
            "total_results": len(result_entries),
            "valid_result_count": len(valid_entries),
            "invalid_result_count": len(result_entries) - len(valid_entries),
            "mean_model_total_valid": (
                round(sum(valid_totals) / len(valid_totals), 4) if valid_totals else None
            ),
            "mean_opponent_total_valid": (
                round(sum(valid_opp_totals) / len(valid_opp_totals), 4) if valid_opp_totals else None
            ),
            "leaderboard_valid_only": leaderboard,
        },
        "results": result_entries,
    }
    summary_path = output_dir / f"groq_robotrumble_summary.{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print_summary(
        results,
        summary_path=summary_path,
        expected_rounds=args.rounds,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
