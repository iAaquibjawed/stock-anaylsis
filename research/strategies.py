"""
Strategy presets (configuration, not code)
==========================================
Ready-made StrategyConfig objects for research. These are *configurations* of the
frozen engine, not new infrastructure — swap one in via the runner to test an idea.

CLASSIC_MOMENTUM is the verification strategy: a well-documented factor whose
qualitative behavior is known from the literature (positive long-run premium,
painful drawdowns / "momentum crashes" after sharp reversals). Use it to sanity-
check the platform on real data before researching novel ideas. Don't expect to
match published returns exactly — universe, costs, and implementation differ — but
results wildly off from the literature mean "audit the data/implementation first".

Import from the engines package, e.g.:
    import sys; sys.path.append("../engines")
    from strategies import CLASSIC_MOMENTUM
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "engines"))
from scoring_engine import StrategyConfig  # noqa: E402


# --- Verification strategy: classic 12-1 momentum, top 20, equal weight ---------
CLASSIC_MOMENTUM = StrategyConfig(
    name="classic_momentum_12_1",
    group_weights={"classic_momentum": 1.0},   # opt in to the 12-1 feature only
    normalize="rank",
    top_n=20,
    min_adv_20=5e7,        # tradeable liquidity floor
    max_vol_20d=None,      # don't filter on vol for the canonical test
)

# --- A few research presets (each is just a different weighting) -----------------
BLENDED_MOMENTUM = StrategyConfig(
    name="blended_momentum",
    group_weights={"momentum": 0.6, "trend": 0.25, "liquidity": 0.15},
    top_n=20,
)

LOW_VOLATILITY = StrategyConfig(
    name="low_volatility",
    group_weights={"volatility": 0.6, "liquidity": 0.25, "trend": 0.15},
    top_n=20,
)

QUALITY_TREND = StrategyConfig(
    name="quality_trend",
    group_weights={"trend": 0.45, "volatility": 0.30, "liquidity": 0.25},
    top_n=20,
)

MEAN_REVERSION = StrategyConfig(
    name="mean_reversion",
    # oscillator group flips usefully here if you invert RSI in a custom config;
    # as a starting point this leans on short-term pullbacks within an uptrend.
    group_weights={"oscillator": 0.5, "trend": 0.3, "liquidity": 0.2},
    top_n=20,
)

PRESETS = {
    s.name: s for s in
    [CLASSIC_MOMENTUM, BLENDED_MOMENTUM, LOW_VOLATILITY, QUALITY_TREND, MEAN_REVERSION]
}
