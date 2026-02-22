# robotrumble-prime

Prime Lab starter environment for RobotRumble-style code generation.

### Overview
- **Environment ID**: `robotrumble-prime`
- **Python package/import name**: `robotrumble_prime`
- **Type**: Single-turn code generation
- **Goal**: Train/evaluate models to produce syntactically valid RobotRumble `robot.py` logic

This package now includes a second module:
- **Environment ID**: `robotrumble-prime-canonical`
- **Module**: `robotrumble_prime_canonical.py`
- **Type**: Multi-turn canonical carryover ladder

### `robotrumble-prime` vs `robotrumble-prime-canonical`
| Aspect | `robotrumble-prime` | `robotrumble-prime-canonical` |
| --- | --- | --- |
| Interaction type | Single-turn (`SingleTurnEnv`) | Multi-turn (`MultiTurnEnv`) |
| Code carryover | No (one-shot candidate code per rollout) | Yes (latest code persists round-to-round) |
| Ladder progression | Fixed-ladder score from one static candidate | Canonical round-by-round progression; advancement gates move to next opponent |
| Stop condition | One model completion | Stops on ladder failure/clear (or max turns) |
| Main use case | Fast RL bring-up, dense/strict reward ablations | Full long-horizon CC:Ladder-style behavior |
| Compute cost | Lower | Higher |

### What It Scores
This environment supports four scoring modes:

1. Bootstrap mode (`split=train` or `split=eval`)
- Fast reward for code-shape bring-up:
- correct `def robot(state, unit):` signature
- exact return contract: `Action.move(Direction.X)` or `Action.attack(Direction.X)` on every return path
- exact RobotRumble API shape (`Action`, `Direction`, `state`, `unit`) with no `Action`/`Direction` shadowing
- Python syntax validity
- avoiding unsafe constructs (`os.system`, `subprocess`, `eval`, `exec`)

2. Ladder mode (`split=ladder_train` or `split=ladder_eval`)
- Adds strict CC:Ladder advancement scoring against:
  - `human/entropicdrifter/seven-of-nine`
  - `human/entropicdrifter/we-are-borg`
  - `human/entropicdrifter/gigachad`
- Uses CC:Ladder-style advancement logic (odd rounds, majority + last-round win).
- Reward is advancement-only (highest ladder progress), with no shaping and no partial within-opponent credit.

3. Bootstrap ladder mode (`split=ladder_bootstrap_train` or `split=ladder_bootstrap_eval`)
- Same rubric family as ladder mode, but uses a single fixed opponent (`seven-of-nine`)
- Useful for cheaper hill-climb bring-up before full 3-opponent ladder runs

4. Stratified ladder-vs-humans eval (`split=ladder_vs_humans_eval`)
- Uses seeded stratified opponent sampling across Elo tiers:
  - Bottom quartile: 2 opponents (`bot1`, `jippty5`)
  - Middle: 3 or 4 opponents sampled from (`maxad`, `alpha_13`, `sivuy`, `bash-brothers`)
  - Top quartile: 2 opponents sampled from (`we-are-borg`, `seven-of-nine`, `gigachad`)
- Opponent sampling is stable by default for cross-run comparability (`ROBOTRUMBLE_STRATIFIED_OPPONENT_SEED=2026`).
- Reward uses dense `ladder_strength` (includes partial progress on the first opponent where the model stalls).
- `ladder_advancement` is logged as monitoring-only.
- Set `ROBOTRUMBLE_STRATIFIED_MIDDLE_COUNT=4` to use 4 middle-tier opponents (default: 3).
- Set `ROBOTRUMBLE_STRATIFIED_OPPONENT_SEED=<int>` only when intentionally rotating the stratified eval set.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"train"` | One of: `train`, `eval`, `ladder_train`, `ladder_eval`, `ladder_bootstrap_train`, `ladder_bootstrap_eval`, `ladder_vs_humans_eval` |
| `max_examples` | int | `-1` | Limit number of examples |
| `seed` | int | `42` | Shuffle seed |

For `ladder_vs_humans_eval`, metadata also records:
- `stratified_middle_count`
- `stratified_opponent_seed`
- `stratified_opponents`

### Reward Note
- `ladder_train`, `ladder_eval`, and bootstrap ladder splits use strict advancement-only reward aligned to CC:Ladder.
- `ladder_vs_humans_eval` uses dense `ladder_strength` on a stratified opponent set.
- Gameplay scoring enforces a strict gate (exact `robot(state, unit)` signature, exact `Action.move/attack(Direction.*)` returns, no API symbol redefinition, syntax-valid, safe code).

### Runtime Note
- Ladder gameplay scoring uses the `rumblebot` binary from the RobotRumble repository.
- It is expected to work in Linux hosted environments.
- On non-Linux local machines, ladder runner readiness may be `0.0` and ladder score may default to `0.0`.

### Local Setup
Install Prime CLI and log in first:

```bash
uv tool install -U prime
prime login
```

Install this local environment from the repo root:

```bash
prime env install robotrumble-prime -p ./environments
```

Run local evaluation:

```bash
prime eval run robotrumble-prime \
  -m qwen/qwen3-30b-a3b-instruct-2507 \
  -n 20 -r 2 \
  -a '{"split":"eval","max_examples":20,"seed":1337}'
```

Run ladder-mode eval:

```bash
prime eval run robotrumble-prime \
  -m qwen/qwen3-30b-a3b-instruct-2507 \
  -n 6 -r 1 \
  -a '{"split":"ladder_eval","max_examples":6,"seed":1337}'
```

Run stratified ladder-vs-humans eval:

```bash
prime eval run robotrumble-prime \
  -m qwen/qwen3-30b-a3b-instruct-2507 \
  -n 6 -r 1 \
  -a '{"split":"ladder_vs_humans_eval","max_examples":6,"seed":1337}'
```

Run canonical carryover eval:

```bash
prime eval run robotrumble-prime-canonical \
  -m qwen/qwen3-30b-a3b-instruct-2507 \
  -n 3 -r 1 \
  -a '{"split":"eval","max_examples":3,"seed":1337,"rounds_per_opponent":5,"turns_per_match":100}'
```

Canonical env defaults:
- Starts from a real human branch codebase (`initial_branch`, default first ladder opponent).
- Uses ordered opponents from weaker to stronger.
- Plays odd `rounds_per_opponent` (default `5`) and requires majority + last-round win to advance.
- Reward is adaptation-aware in training:
  - Train split blends dense `canonical_ladder_strength`, strict `canonical_ladder_advancement`,
    tie-aware `canonical_competitive_score`, `canonical_late_round_quality`, and `canonical_round_win_rate`.
  - Contract metrics are tracked (`canonical_contract_success_rate`, `canonical_contract_failure_rate`) but
    are monitoring-only in train shaping to avoid collapsing to a legality-only reward floor.
  - Eval split is aligned to dense `canonical_ladder_strength` for cleaner promotion gating.
  - `canonical_contract_failure_rate` remains logged as a monitoring metric.
- Turn prompts enforce targeted round-to-round patches (not full rewrites) and present current code in
  `<robot_py>...</robot_py>` blocks to reduce markdown-fence leakage.
- Hosted RL note: when using the same Hub slug (`codeclash/robotrumble-prime`), set `args.variant="canonical"` in both `[[env]]` and `[[eval.env]]` to route to canonical MultiTurn logic.

### Push To Hub (for Hosted Training)

```bash
prime env push -p ./environments/robotrumble_prime -v PRIVATE
```

After push, replace local env id in training config with your hub id (for example `your_org/robotrumble-prime`).
