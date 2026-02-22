"""Pytest compatibility plugin for Prime hub integration filtering.

This only activates inside the `research-environments` harness used by
Prime's environment integration job. It tags collected tests with
`robotrumble_prime` so selector mismatches do not lead to zero selected tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _is_prime_research_harness(rootpath: Path) -> bool:
    return rootpath.name == "research-environments"


def pytest_configure(config):
    rootpath = Path(str(config.rootpath))
    if not _is_prime_research_harness(rootpath):
        return
    sys.stderr.write(
        f"[robotrumble_prime_plugin] configure keyword={config.option.keyword!r} markexpr={config.option.markexpr!r}\n"
    )
    # Prime integration sometimes applies a selector that mismatches collected
    # test ids in the upstream harness, yielding "0 selected" (exit 5).
    # This job is already scoped to a single changed env (4 tests total), so
    # clearing that selector safely restores intended execution.
    keyword_expr = str(config.option.keyword or "")
    if "robotrumble_prime" in keyword_expr or "robotrumble-prime" in keyword_expr:
        config.option.keyword = ""
        sys.stderr.write(
            f"[robotrumble_prime_plugin] keyword normalized to={config.option.keyword!r}\n"
        )
    config.addinivalue_line(
        "markers",
        "robotrumble_prime: compatibility marker injected by robotrumble_prime env package",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_itemcollected(item):
    rootpath = Path(str(item.config.rootpath))
    if not _is_prime_research_harness(rootpath):
        return
    if getattr(item.config, "_robotrumble_prime_plugin_debug_seen", False) is False:
        item.config._robotrumble_prime_plugin_debug_seen = True
        sys.stderr.write(
            f"[robotrumble_prime_plugin] first_item nodeid={item.nodeid}\n"
        )
    item.add_marker("robotrumble_prime")
    item.keywords["robotrumble_prime"] = True


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    rootpath = Path(str(config.rootpath))
    if not _is_prime_research_harness(rootpath):
        return
    for item in items:
        item.add_marker("robotrumble_prime")
        item.keywords["robotrumble_prime"] = True
