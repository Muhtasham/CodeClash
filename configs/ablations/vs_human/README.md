# vs. Human

These set of configurations correspond to Section 4.1 of the original paper, specifically the subsection *On RobotRumble, models trail substantially behind expert human programmers*.

Each configuration pits a model against an open source codebase written by a human expert for a particular arena. Across a tournament spanning 15 rounds, the model is allowed the evolve the codebase as it sees fit to beat the human expert's solution. The human's solution is *not* changing for the duration of the tournament.

To make models compete against static human solutions, do the following two steps.

1. Make sure the human solution is working and pushed as a branch to the corresponding arena. E.g. [gigachad](https://github.com/CodeClash-ai/RobotRumble/tree/human/entropicdrifter/gigachad) for RobotRumble.
2. Then, in your configuration, simply specify one of the players as a `dummy` agent, with `branch_init` set to the branch name, such as:

```yaml
players:
- agent: dummy
  branch_init: human/entropicdrifter/gigachad
  name: gigachad
```

## Groq RobotRumble benchmark workflow

Use the helper script to benchmark multiple Groq-hosted models against the same human baseline on RobotRumble:

```bash
uv run python scripts/benchmark_robotrumble_groq.py \
  --rounds 3 \
  --sims-per-round 250 \
  --opponent-branch human/entropicdrifter/seven-of-nine
```

Notes:
- Requires `GROQ_API_KEY` in your environment.
- Generates per-model config files under `configs/ablations/vs_human/generated_groq/`.
- Writes tournament logs and a summary JSON under `logs/groq_robotrumble/`.
- Groq Chat API rejects unsupported message fields (for example `timestamp` / `extra`). CodeClash keeps those in trajectory logs, and sends API-safe messages.

To discover active Groq models from the API first:

```bash
uv run python scripts/benchmark_robotrumble_groq.py --discover-models --list-models-only
```

## Prime eval note (`robotrumble-prime`)

The `ladder_vs_humans_eval` split now uses seeded stratified sampling across human-opponent strength tiers:
- 2 bottom-tier opponents (`bot1`, `jippty5`)
- 3 (default) or 4 middle-tier opponents sampled from (`maxad`, `alpha_13`, `sivuy`, `bash-brothers`)
- 2 top-tier opponents sampled from (`we-are-borg`, `seven-of-nine`, `gigachad`)
- Sampling is stable by default for comparability (`ROBOTRUMBLE_STRATIFIED_OPPONENT_SEED=2026`).

Reward for this split is dense `ladder_strength`, with strict `ladder_advancement` retained as a monitoring metric.
Set `ROBOTRUMBLE_STRATIFIED_MIDDLE_COUNT=4` to run with four middle-tier opponents.
Set `ROBOTRUMBLE_STRATIFIED_OPPONENT_SEED=<int>` only when intentionally evaluating on a different stratified set.
