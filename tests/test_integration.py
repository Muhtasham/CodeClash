"""
Integration tests for all game arenas.

These tests verify that the main execution flow works without exceptions,
using DeterministicModel instead of real LLM models.
"""

from codeclash import CONFIG_DIR


# ============================================================================
# BattleSnake Tests
# ============================================================================


def test_pvp_battlesnake():
    """Test BattleSnake game in PvP mode."""
    from main import main_cli

    config_path = CONFIG_DIR / "test" / "battlesnake_pvp_test.yaml"
    main_cli(["-c", str(config_path)])


def test_single_player_battlesnake():
    """Test BattleSnake game in single player mode."""
    from scripts.main_single_player import main_cli

    config_path = CONFIG_DIR / "test" / "battlesnake_single_player_test.yaml"
    main_cli(["-c", str(config_path)])


# ============================================================================
# Dummy Tests
# ============================================================================


def test_pvp_dummy():
    """Test Dummy game (simplest game arena)."""
    from main import main_cli

    config_path = CONFIG_DIR / "test" / "dummy.yaml"
    main_cli(["-c", str(config_path)])


# ============================================================================
# RobotRumble Tests
# ============================================================================


def test_pvp_robotrumble():
    """Test RobotRumble game."""
    from main import main_cli

    config_path = CONFIG_DIR / "test" / "robotrumble.yaml"
    main_cli(["-c", str(config_path)])


# ============================================================================
# Halite Tests
# ============================================================================


def test_pvp_halite():
    """Test Halite (original) game."""
    from main import main_cli

    config_path = CONFIG_DIR / "test" / "halite.yaml"
    main_cli(["-c", str(config_path)])


# NOTE: Halite II and Halite III are WIP and not yet implemented in ARENAS
# def test_pvp_halite2():
#     """Test Halite II game."""
#     from main import main_cli
#
#     config_path = CONFIG_DIR / "test" / "halite2.yaml"
#     main_cli(["-c", str(config_path)])
#
#
# def test_pvp_halite3():
#     """Test Halite III game."""
#     from main import main_cli
#
#     config_path = CONFIG_DIR / "test" / "halite3.yaml"
#     main_cli(["-c", str(config_path)])


# ============================================================================
# BattleCode Tests
# ============================================================================


def test_pvp_battlecode():
    """Test BattleCode game."""
    from main import main_cli

    config_path = CONFIG_DIR / "test" / "battlecode.yaml"
    main_cli(["-c", str(config_path)])


# ============================================================================
# CoreWar Tests
# ============================================================================


def test_pvp_corewar():
    """Test CoreWar game."""
    from main import main_cli

    config_path = CONFIG_DIR / "test" / "corewar.yaml"
    main_cli(["-c", str(config_path)])


# ============================================================================
# RoboCode Tests
# ============================================================================


def test_pvp_robocode():
    """Test RoboCode game."""
    from main import main_cli

    config_path = CONFIG_DIR / "test" / "robocode.yaml"
    main_cli(["-c", str(config_path)])


# ============================================================================
# HuskyBench Tests
# ============================================================================


def test_pvp_huskybench():
    """Test HuskyBench game."""
    from main import main_cli

    config_path = CONFIG_DIR / "test" / "huskybench.yaml"
    main_cli(["-c", str(config_path)])
