import argparse
import contextlib
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from agent.ExchangeAgent import ExchangeAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from agent.TradingAgent import TradingAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from Kernel import Kernel
from util import util
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from util.order import LimitOrder

CRASH_SCORE = -1_000_000.0
SYMBOL = "JPM"
STARTING_CASH = 10_000_000
PLAYER_LAMBDA_A = 7e-11
GUARDED_METHOD_DEFAULTS = {
    "kernelInitializing": None,
    "kernelStarting": None,
    "kernelStopping": None,
    "wakeup": False,
    "receiveMessage": None,
}


def safe_module_name(player_name: str, sim_idx: int | None = None) -> str:
    safe = re.sub(r"\W+", "_", player_name)
    if not safe or safe[0].isdigit():
        safe = f"player_{safe}"
    suffix = "" if sim_idx is None else f"_sim_{sim_idx}"
    return f"codeclash_abides_{safe.lower()}{suffix}"


def load_agent_class(player_name: str, path: str, *, sim_idx: int | None = None):
    agent_dir = str(Path(path).resolve().parent)
    sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location(safe_module_name(player_name, sim_idx), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(agent_dir)
    if not hasattr(module, "MyAgent"):
        raise RuntimeError(f"{path} does not define MyAgent")
    agent_class = module.MyAgent
    if not issubclass(agent_class, TradingAgent):
        raise RuntimeError(f"{path} MyAgent must inherit from ABIDES TradingAgent")
    return agent_class


def make_random_state() -> np.random.RandomState:
    return np.random.RandomState(seed=np.random.randint(low=0, high=2**32, dtype="uint64"))


def make_player_agent(agent_class: type, player_name: str, agent_id: int):
    agent = agent_class(
        id=agent_id,
        name=player_name,
        type=f"PLAYER:{player_name}",
        symbol=SYMBOL,
        starting_cash=STARTING_CASH,
        log_orders=False,
        random_state=make_random_state(),
    )
    if hasattr(agent, "lambda_a"):
        agent.lambda_a = min(float(agent.lambda_a), PLAYER_LAMBDA_A)
    return guard_player_agent(agent)


def guard_player_agent(agent: TradingAgent) -> TradingAgent:
    agent._codeclash_error = None
    agent._codeclash_traceback = None

    for method_name, fallback in GUARDED_METHOD_DEFAULTS.items():
        original = getattr(agent, method_name, None)
        if original is None:
            continue

        def guarded(*args, _original=original, _method_name=method_name, _fallback=fallback, **kwargs):
            if getattr(agent, "_codeclash_error", None):
                return _fallback
            try:
                return _original(*args, **kwargs)
            except Exception as exc:
                agent._codeclash_error = f"{type(exc).__name__} in {_method_name}: {exc}"
                agent._codeclash_traceback = traceback.format_exc(limit=5)
                return _fallback

        setattr(agent, method_name, guarded)

    return agent


def make_world_agents(agent_classes: dict[str, type], *, sim_idx: int, market_minutes: int, background_agents: int):
    historical_date = pd.to_datetime("2019-06-28")
    mkt_open = historical_date + pd.to_timedelta("09:30:00")
    mkt_close = mkt_open + pd.to_timedelta(market_minutes, unit="m")
    symbols = {
        SYMBOL: {
            "r_bar": 100000,
            "kappa": 1.67e-12,
            "agent_kappa": 1.67e-15,
            "sigma_s": 0,
            "fund_vol": 1e-8,
            "megashock_lambda_a": 2.77778e-13,
            "megashock_mean": 1e3,
            "megashock_var": 5e4,
            "random_state": make_random_state(),
        }
    }
    oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

    agents = []
    agent_id = 0
    exchange = ExchangeAgent(
        id=agent_id,
        name="EXCHANGE_AGENT",
        type="ExchangeAgent",
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        symbols=[SYMBOL],
        log_orders=False,
        book_freq=None,
        pipeline_delay=0,
        computation_delay=0,
        stream_history=10,
        random_state=make_random_state(),
    )
    agents.append(exchange)
    agent_id += 1

    agents.append(
        MarketMakerAgent(
            id=agent_id,
            name="MARKET_MAKER_AGENT",
            type="MarketMakerAgent",
            symbol=SYMBOL,
            starting_cash=STARTING_CASH,
            min_size=20,
            max_size=50,
            wake_up_freq="30s",
            log_orders=False,
            random_state=make_random_state(),
        )
    )
    agent_id += 1

    symbol_config = symbols[SYMBOL]
    for idx in range(background_agents):
        agents.append(
            ZeroIntelligenceAgent(
                id=agent_id,
                name=f"ZI_AGENT_{idx}",
                type="ZeroIntelligenceAgent",
                symbol=SYMBOL,
                starting_cash=STARTING_CASH,
                sigma_n=10000,
                r_bar=symbol_config["r_bar"],
                kappa=symbol_config["agent_kappa"],
                sigma_s=symbol_config["fund_vol"],
                q_max=10,
                sigma_pv=5e4,
                R_min=0,
                R_max=100,
                eta=1,
                lambda_a=1e-10,
                log_orders=False,
                random_state=make_random_state(),
            )
        )
        agent_id += 1

    player_agents = {}
    player_ids = {}
    player_names = list(agent_classes.keys())
    if player_names:
        offset = sim_idx % len(player_names)
        ordered_names = player_names[offset:] + player_names[:offset]
        for player_name in ordered_names:
            agent = make_player_agent(agent_classes[player_name], player_name, agent_id)
            agents.append(agent)
            player_agents[player_name] = agent
            player_ids[agent_id] = player_name
            agent_id += 1

    for agent in agents:
        agent.log_to_file = False

    ledgers = {player: {"CASH": STARTING_CASH, SYMBOL: 0} for player in player_agents}
    original_exchange_send = exchange.sendMessage

    def scored_exchange_send(*args, **kwargs):
        recipient_id = args[0] if args else kwargs.get("recipientID", kwargs.get("recipient_id"))
        msg = args[1] if len(args) > 1 else kwargs.get("msg")
        player = player_ids.get(recipient_id)
        if player and getattr(msg, "body", {}).get("msg") == "ORDER_EXECUTED":
            order = msg.body["order"]
            quantity = int(order.quantity)
            signed_quantity = quantity if order.is_buy_order else -quantity
            ledgers[player][SYMBOL] += signed_quantity
            ledgers[player]["CASH"] -= signed_quantity * int(order.fill_price)
        return original_exchange_send(*args, **kwargs)

    exchange.sendMessage = scored_exchange_send

    return {
        "agents": agents,
        "player_agents": player_agents,
        "ledgers": ledgers,
        "exchange": exchange,
        "historical_date": historical_date,
        "mkt_close": mkt_close,
        "oracle": oracle,
    }


def score_player(agent: TradingAgent, ledger: dict[str, int], final_price: int) -> tuple[float, dict]:
    if getattr(agent, "_codeclash_error", None):
        return CRASH_SCORE, {
            "status": "error",
            "error": agent._codeclash_error,
            "traceback": agent._codeclash_traceback,
        }

    try:
        cash = int(ledger.get("CASH", 0))
        shares = int(ledger.get(SYMBOL, 0))
        score = float(cash + shares * final_price - STARTING_CASH)
    except Exception as exc:
        return CRASH_SCORE, {
            "status": "error",
            "error": f"{type(exc).__name__} while scoring: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }

    return score, {"status": "ok", "cash": cash, "shares": shares}


def run_player_market(
    player: str,
    agent_class: type,
    *,
    sim_idx: int,
    market_minutes: int,
    background_agents: int,
) -> dict:
    seed = 9200 + sim_idx
    random.seed(seed)
    np.random.seed(seed)
    util.silent_mode = True
    LimitOrder.silent_mode = True

    world = make_world_agents(
        {player: agent_class},
        sim_idx=sim_idx,
        market_minutes=market_minutes,
        background_agents=background_agents,
    )
    kernel = Kernel("CodeClash ABIDES Kernel", random_state=make_random_state())
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.devnull, "w") as devnull, contextlib.chdir(tmpdir), contextlib.redirect_stdout(devnull):
            kernel.runner(
                agents=world["agents"],
                startTime=world["historical_date"],
                stopTime=world["mkt_close"] + pd.to_timedelta("00:01:00"),
                defaultComputationDelay=50,
                defaultLatency=1,
                oracle=world["oracle"],
                skip_log=True,
                log_dir=None,
            )

    final_price = int(world["exchange"].order_books[SYMBOL].last_trade)
    agent = world["player_agents"][player]
    ledger = world["ledgers"][player]
    score, score_detail = score_player(agent, ledger, final_price)
    return {
        "score": score,
        "detail": {
            "sim": sim_idx,
            "player": player,
            "score": score,
            "final_price": final_price,
            **score_detail,
        },
    }


def run_player_market_worker(
    result_queue: mp.Queue,
    player: str,
    path: str,
    *,
    sim_idx: int,
    market_minutes: int,
    background_agents: int,
) -> None:
    try:
        agent_class = load_agent_class(player, path, sim_idx=sim_idx)
        result_queue.put(
            run_player_market(
                player, agent_class, sim_idx=sim_idx, market_minutes=market_minutes, background_agents=background_agents
            )
        )
    except BaseException as exc:
        result_queue.put(
            {
                "score": CRASH_SCORE,
                "detail": {
                    "sim": sim_idx,
                    "player": player,
                    "score": CRASH_SCORE,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=5),
                },
            }
        )


def run_player_market_isolated(
    player: str,
    path: str,
    *,
    sim_idx: int,
    market_minutes: int,
    background_agents: int,
    player_timeout: int,
) -> dict:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=run_player_market_worker,
        args=(result_queue, player, path),
        kwargs={
            "sim_idx": sim_idx,
            "market_minutes": market_minutes,
            "background_agents": background_agents,
        },
    )
    process.start()
    process.join(player_timeout)
    if process.is_alive():
        process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            "score": CRASH_SCORE,
            "detail": {
                "sim": sim_idx,
                "player": player,
                "score": CRASH_SCORE,
                "status": "error",
                "error": f"player simulation exceeded {player_timeout}s timeout",
            },
        }

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            "score": CRASH_SCORE,
            "detail": {
                "sim": sim_idx,
                "player": player,
                "score": CRASH_SCORE,
                "status": "error",
                "error": f"player simulation exited with code {process.exitcode} and no result",
            },
        }


def run_market(
    agent_paths: dict[str, str], *, sim_idx: int, market_minutes: int, background_agents: int, player_timeout: int
) -> dict:
    scores = {}
    details = []
    for player, path in agent_paths.items():
        result = run_player_market_isolated(
            player,
            path,
            sim_idx=sim_idx,
            market_minutes=market_minutes,
            background_agents=background_agents,
            player_timeout=player_timeout,
        )
        scores[player] = result["score"]
        details.append(result["detail"])

    return {"scores": scores, "details": details}


def parse_agent_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--agent values must be NAME=/path/to/abides_agent.py")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("agent name cannot be empty")
    if not Path(path).exists():
        raise argparse.ArgumentTypeError(f"agent path does not exist: {path}")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="append", type=parse_agent_arg, required=True)
    parser.add_argument("--sims", type=int, default=3)
    parser.add_argument("--market-minutes", type=int, default=5)
    parser.add_argument("--background-agents", type=int, default=3)
    parser.add_argument("--player-timeout", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.sims < 1:
        parser.error("--sims must be at least 1")
    if args.market_minutes < 1:
        parser.error("--market-minutes must be at least 1")
    if args.background_agents < 0:
        parser.error("--background-agents cannot be negative")
    if args.player_timeout < 1:
        parser.error("--player-timeout must be at least 1")

    agent_names = [name for name, _ in args.agent]
    if len(agent_names) != len(set(agent_names)):
        parser.error("--agent names must be unique")

    agent_paths = dict(args.agent)
    totals = {name: 0.0 for name in agent_paths}
    details = []

    for sim_idx in range(args.sims):
        try:
            result = run_market(
                agent_paths,
                sim_idx=sim_idx,
                market_minutes=args.market_minutes,
                background_agents=args.background_agents,
                player_timeout=args.player_timeout,
            )
        except Exception as exc:
            result = {
                "scores": {name: CRASH_SCORE for name in agent_paths},
                "details": [
                    {
                        "sim": sim_idx,
                        "player": name,
                        "score": CRASH_SCORE,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=5),
                    }
                    for name in agent_paths
                ],
            }
        for player, score in result["scores"].items():
            totals[player] += score
        details.extend(result["details"])

    averages = {player: score / args.sims for player, score in totals.items()}
    output = {
        "average_scores": averages,
        "total_scores": totals,
        "sims": args.sims,
        "details": [json.dumps(item, sort_keys=True) for item in details],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
