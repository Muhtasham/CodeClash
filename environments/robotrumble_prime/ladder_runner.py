from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROBOTRUMBLE_REPO_URL = "https://github.com/CodeClash-ai/RobotRumble.git"
GITHUB_RAW_PREFIX = "https://raw.githubusercontent.com/CodeClash-ai/RobotRumble"

DEFAULT_FIXED_OPPONENTS: tuple[str, ...] = (
    "human/entropicdrifter/seven-of-nine",
    "human/entropicdrifter/we-are-borg",
    "human/entropicdrifter/gigachad",
)

_ASSET_DIR = Path(tempfile.gettempdir()) / "robotrumble_prime_assets"
_REPO_DIR = _ASSET_DIR / "RobotRumble"
_RUMBLEBOT_PATH = _REPO_DIR / "rumblebot"
_OPPONENTS_DIR = _ASSET_DIR / "opponents"
_CANDIDATES_DIR = _ASSET_DIR / "candidates"

_LOCK = threading.Lock()
_ASSETS_READY = False
_ASSETS_ERROR: str | None = None

_OPPONENT_FILE_CACHE: dict[str, Path] = {}
_LADDER_CACHE: dict[str, LadderEvalResult] = {}


@dataclass(frozen=True)
class OpponentEval:
    branch: str
    wins: int
    losses: int
    ties: int
    majority_win: bool
    won_last_round: bool
    advanced: bool


@dataclass(frozen=True)
class LadderEvalResult:
    available: bool
    error: str | None
    score: float
    defeated_count: int
    opponents_evaluated: tuple[OpponentEval, ...]


@dataclass(frozen=True)
class SingleMatchResult:
    available: bool
    error: str | None
    outcome: str


def _run_subprocess(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _ensure_assets() -> tuple[bool, str | None]:
    global _ASSETS_READY, _ASSETS_ERROR
    if _ASSETS_READY:
        return True, None
    if _ASSETS_ERROR is not None:
        return False, _ASSETS_ERROR

    with _LOCK:
        if _ASSETS_READY:
            return True, None
        if _ASSETS_ERROR is not None:
            return False, _ASSETS_ERROR

        _ASSET_DIR.mkdir(parents=True, exist_ok=True)
        _OPPONENTS_DIR.mkdir(parents=True, exist_ok=True)
        _CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

        try:
            if not _REPO_DIR.exists():
                clone_result = _run_subprocess(
                    ["git", "clone", "--depth", "1", ROBOTRUMBLE_REPO_URL, str(_REPO_DIR)],
                    timeout=180,
                )
                if clone_result.returncode != 0:
                    raise RuntimeError(clone_result.stderr.strip() or clone_result.stdout.strip())

            if not _RUMBLEBOT_PATH.exists():
                raise FileNotFoundError(f"Missing binary at {_RUMBLEBOT_PATH}")

            _RUMBLEBOT_PATH.chmod(0o755)
            health = _run_subprocess([str(_RUMBLEBOT_PATH), "--help"], cwd=_REPO_DIR, timeout=15)
            if health.returncode != 0:
                msg = health.stderr.strip() or health.stdout.strip()
                raise RuntimeError(f"rumblebot health check failed: {msg}")

            _ASSETS_READY = True
            _ASSETS_ERROR = None
            return True, None
        except Exception as exc:
            _ASSETS_READY = False
            _ASSETS_ERROR = f"{type(exc).__name__}: {exc}"
            return False, _ASSETS_ERROR


def runner_status() -> tuple[bool, str | None]:
    return _ensure_assets()


def _download_text(url: str) -> str | None:
    request = Request(
        url,
        headers={
            "User-Agent": "codeclash-robotrumble-prime-v0.2",
            "Accept": "text/plain",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except URLError:
        return None


def _opponent_cache_key(branch: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", branch.strip().lower())


def _get_opponent_file(branch: str) -> Path:
    if branch in _OPPONENT_FILE_CACHE:
        return _OPPONENT_FILE_CACHE[branch]

    key = _opponent_cache_key(branch)
    for ext in ("py", "js"):
        url = f"{GITHUB_RAW_PREFIX}/{branch}/robot.{ext}"
        source = _download_text(url)
        if source:
            path = _OPPONENTS_DIR / f"{key}.robot.{ext}"
            path.write_text(source)
            _OPPONENT_FILE_CACHE[branch] = path
            return path

    raise FileNotFoundError(f"Could not fetch robot.py/js for branch '{branch}'")


def _candidate_file(code: str) -> Path:
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    path = _CANDIDATES_DIR / f"candidate_{code_hash[:16]}.py"
    if not path.exists():
        path.write_text(code)
    return path


def _parse_match_json(raw_output: str) -> dict:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    try:
        parsed = json.loads(raw_output.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    raise ValueError("Could not parse rumblebot JSON output")


def _run_match(
    candidate_path: Path,
    opponent_path: Path,
    *,
    turns_per_match: int,
    seed: str,
    timeout_seconds: int,
) -> str:
    cmd = [
        str(_RUMBLEBOT_PATH),
        "run",
        "term",
        str(candidate_path),
        str(opponent_path),
        "--raw",
        "-t",
        str(turns_per_match),
        "--seed",
        seed,
    ]
    result = _run_subprocess(cmd, cwd=_REPO_DIR, timeout=timeout_seconds)
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"rumblebot exited with {result.returncode}: {msg}")

    payload = _parse_match_json(result.stdout)
    winner = payload.get("winner")
    if winner == "Blue":
        return "win"
    if winner == "Red":
        return "loss"
    return "tie"


def _cache_key(
    code: str,
    opponents: Sequence[str],
    rounds_per_opponent: int,
    turns_per_match: int,
    base_seed: int,
) -> str:
    payload = {
        "code_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "opponents": list(opponents),
        "rounds_per_opponent": rounds_per_opponent,
        "turns_per_match": turns_per_match,
        "base_seed": base_seed,
    }
    return json.dumps(payload, sort_keys=True)


def get_opponent_source(branch: str) -> tuple[bool, str | None, str | None]:
    available, error = _ensure_assets()
    if not available:
        return False, None, error
    try:
        path = _get_opponent_file(branch)
        return True, path.read_text(), None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def run_single_match(
    candidate_code: str,
    *,
    opponent: str,
    turns_per_match: int = 100,
    seed: str = "1337-0-0",
    timeout_seconds: int = 120,
) -> SingleMatchResult:
    available, error = _ensure_assets()
    if not available:
        return SingleMatchResult(available=False, error=error, outcome="loss")

    try:
        candidate_path = _candidate_file(candidate_code)
        opponent_path = _get_opponent_file(opponent)
        outcome = _run_match(
            candidate_path,
            opponent_path,
            turns_per_match=turns_per_match,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        return SingleMatchResult(available=True, error=None, outcome=outcome)
    except Exception as exc:
        return SingleMatchResult(
            available=False,
            error=f"{type(exc).__name__}: {exc}",
            outcome="loss",
        )


def evaluate_fixed_ladder(
    candidate_code: str,
    *,
    opponents: Sequence[str] = DEFAULT_FIXED_OPPONENTS,
    rounds_per_opponent: int = 3,
    turns_per_match: int = 100,
    base_seed: int = 1337,
    timeout_seconds: int = 120,
) -> LadderEvalResult:
    if rounds_per_opponent < 3 or rounds_per_opponent % 2 == 0:
        raise ValueError("rounds_per_opponent must be odd and >= 3")
    if not opponents:
        raise ValueError("opponents must be non-empty")

    cache_key = _cache_key(candidate_code, opponents, rounds_per_opponent, turns_per_match, base_seed)
    if cache_key in _LADDER_CACHE:
        return _LADDER_CACHE[cache_key]

    available, error = _ensure_assets()
    if not available:
        result = LadderEvalResult(
            available=False,
            error=error,
            score=0.0,
            defeated_count=0,
            opponents_evaluated=tuple(),
        )
        _LADDER_CACHE[cache_key] = result
        return result

    try:
        candidate_path = _candidate_file(candidate_code)
        opponent_results: list[OpponentEval] = []
        defeated = 0

        for opp_idx, branch in enumerate(opponents):
            opp_path = _get_opponent_file(branch)
            wins = 0
            losses = 0
            ties = 0
            outcomes: list[str] = []

            for round_idx in range(rounds_per_opponent):
                seed = f"{base_seed}-{opp_idx}-{round_idx}"
                outcome = _run_match(
                    candidate_path,
                    opp_path,
                    turns_per_match=turns_per_match,
                    seed=seed,
                    timeout_seconds=timeout_seconds,
                )
                outcomes.append(outcome)
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
                else:
                    ties += 1

            majority_win = wins >= (rounds_per_opponent // 2 + 1)
            won_last_round = outcomes[-1] == "win"
            advanced = majority_win and won_last_round

            opponent_results.append(
                OpponentEval(
                    branch=branch,
                    wins=wins,
                    losses=losses,
                    ties=ties,
                    majority_win=majority_win,
                    won_last_round=won_last_round,
                    advanced=advanced,
                )
            )

            if advanced:
                defeated += 1
                continue
            break

        partial = 0.0
        if defeated < len(opponent_results):
            current = opponent_results[defeated]
            partial = current.wins / float(rounds_per_opponent)

        score = (defeated + partial) / float(len(opponents))
        result = LadderEvalResult(
            available=True,
            error=None,
            score=max(0.0, min(score, 1.0)),
            defeated_count=defeated,
            opponents_evaluated=tuple(opponent_results),
        )
        _LADDER_CACHE[cache_key] = result
        return result
    except Exception as exc:
        result = LadderEvalResult(
            available=False,
            error=f"{type(exc).__name__}: {exc}",
            score=0.0,
            defeated_count=0,
            opponents_evaluated=tuple(),
        )
        _LADDER_CACHE[cache_key] = result
        return result
