"""
Scoring Engine (Engine 3)
=========================
Turns a cross-sectional feature table (one row per stock, as of a rebalance
date) into a ranked shortlist. Fully GENERIC and config-driven:

  - Weights are NOT hard-coded. A StrategyConfig declares group weights, the
    normalization method, and risk filters. Swap the config -> swap the strategy
    (momentum, quality, mean-reversion, ...) with zero code changes.
  - Cross-sectional NORMALIZATION: each feature is converted to a relative
    measure (percentile rank or winsorized z-score), so RSI=68 becomes
    "92nd percentile", and features on different scales combine sanely.
  - DIRECTION-AWARE: lower_better features (volatility, gap) are flipped so a
    high normalized value always means "better".
  - RISK FILTERS applied BEFORE ranking: illiquid / too-volatile / penny names
    are removed so they can't win on a noisy signal.

The engine reads feature semantics from feature_engine.FEATURE_META, so it never
needs to know what a column means — only the registry does.

Look-ahead safety: scoring operates on a single cross-section produced by
feature_engine.cross_section(symbols, date), which is itself causal. The scorer
adds no time dimension and therefore introduces no leakage.

Research scaffolding, not investment advice. Output is a shortlist to research.

Dependencies:
    pip install pandas numpy
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import feature_engine as fe


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    """Everything that defines a strategy, in data not code."""
    name: str = "balanced_momentum"
    # Weights per feature GROUP (from FEATURE_META). Need not sum to 1 — they
    # are normalized internally. Groups absent here get weight 0.
    group_weights: dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.30,
        "trend": 0.25,
        "liquidity": 0.20,
        "volatility": 0.15,
        "oscillator": 0.10,
    })
    normalize: str = "rank"          # "rank" (percentile 0..1) or "zscore"
    winsor: float = 0.02             # clip tails before zscore (each side)
    top_n: int = 5
    # Risk filters (applied before ranking). None disables a filter.
    min_adv_20: float | None = 5e7   # min avg daily value traded (₹), liquidity
    max_vol_20d: float | None = 0.80 # max annualized volatility
    min_price: float | None = None   # optional penny-stock floor (uses 'price')
    require_uptrend: bool = False     # if True, keep only golden==1


# ---------------------------------------------------------------------------
# Normalization (cross-sectional)
# ---------------------------------------------------------------------------
def _winsorize(s: pd.Series, p: float) -> pd.Series:
    if p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def _normalize_column(s: pd.Series, method: str, winsor: float) -> pd.Series:
    """Map a raw feature column to a 0..1-ish 'better is higher' scale."""
    s = s.astype(float)
    if s.notna().sum() <= 1:
        return pd.Series(0.5, index=s.index)  # nothing to rank against
    if method == "rank":
        return s.rank(pct=True)               # percentile in [0,1]
    if method == "zscore":
        s = _winsorize(s, winsor)
        mu, sd = s.mean(), s.std()
        if not sd or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sd
    raise ValueError(f"unknown normalize method: {method}")


def normalize_cross_section(cs: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Return a frame of normalized, direction-corrected feature scores."""
    norm = pd.DataFrame(index=cs.index)
    for col, meta in fe.FEATURE_META.items():
        if col not in cs.columns:
            continue
        direction = meta["direction"]
        method = meta.get("normalize", cfg.normalize)
        if method == "none":
            val = cs[col].astype(float)        # already 0/1 flag
        else:
            val = _normalize_column(cs[col], cfg.normalize, cfg.winsor)
        if direction == "lower_better":
            # flip: for rank -> 1-pct; for zscore -> negate
            val = (1.0 - val) if cfg.normalize == "rank" and method != "none" else -val
        norm[col] = val
    return norm


# ---------------------------------------------------------------------------
# Risk filters
# ---------------------------------------------------------------------------
def apply_risk_filters(cs: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, dict]:
    """Drop rows failing hard risk constraints. Returns (kept, drop_counts)."""
    keep = pd.Series(True, index=cs.index)
    drops: dict[str, int] = {}

    def _apply(mask: pd.Series, label: str):
        nonlocal keep
        failed = ~mask.reindex(cs.index).fillna(False)
        # treat NaN feature as failing the filter (conservative)
        drops[label] = int((keep & failed).sum())
        keep = keep & ~failed

    if cfg.min_adv_20 is not None and "adv_20" in cs:
        _apply(cs["adv_20"] >= cfg.min_adv_20, "illiquid")
    if cfg.max_vol_20d is not None and "vol_20d" in cs:
        _apply(cs["vol_20d"] <= cfg.max_vol_20d, "too_volatile")
    if cfg.min_price is not None and "price" in cs:
        _apply(cs["price"] >= cfg.min_price, "penny")
    if cfg.require_uptrend and "golden" in cs:
        _apply(cs["golden"] == 1.0, "not_uptrend")

    return cs[keep], drops


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _group_scores(norm: pd.DataFrame) -> pd.DataFrame:
    """Average the normalized features within each group -> one col per group."""
    groups: dict[str, list[str]] = {}
    for col in norm.columns:
        g = fe.FEATURE_META[col]["group"]
        groups.setdefault(g, []).append(col)
    gs = pd.DataFrame(index=norm.index)
    for g, cols in groups.items():
        gs[g] = norm[cols].mean(axis=1, skipna=True)
    return gs


def score(cs: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    """
    Full pipeline: risk filter -> normalize -> group scores -> weighted total
    -> rank. Returns a DataFrame sorted best-first with score + group breakdown.
    """
    cfg = cfg or StrategyConfig()
    kept, drops = apply_risk_filters(cs, cfg)
    if kept.empty:
        return pd.DataFrame(columns=["score"])

    norm = normalize_cross_section(kept, cfg)
    gs = _group_scores(norm)

    # Normalize weights over groups that actually exist
    weights = {g: w for g, w in cfg.group_weights.items() if g in gs.columns}
    wsum = sum(weights.values()) or 1.0
    weights = {g: w / wsum for g, w in weights.items()}

    total = pd.Series(0.0, index=gs.index)
    for g, w in weights.items():
        total = total + w * gs[g].fillna(gs[g].mean())

    out = gs.copy()
    out["score"] = total
    out = out.sort_values("score", ascending=False)
    out.attrs["drops"] = drops
    out.attrs["weights"] = weights
    return out


def rank_universe(symbols: list[str], date, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    """Convenience: build the cross-section from the Feature Engine, then score."""
    cfg = cfg or StrategyConfig()
    cs = fe.cross_section(symbols, date)
    ranked = score(cs, cfg)
    return ranked.head(cfg.top_n)


# ---------------------------------------------------------------------------
# Factor attribution — WHY each stock scored what it did
# ---------------------------------------------------------------------------
def contributions(ranked: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Per-stock weighted contribution of each group to its total score.
    Rows sum (across group columns) to the 'score' column. This is the
    factor attribution: if a strategy breaks, you see which factor moved.
    """
    weights = weights or ranked.attrs.get("weights", {})
    groups = [g for g in weights if g in ranked.columns]
    contrib = ranked[groups].mul(pd.Series(weights)[groups], axis=1)
    contrib["total"] = contrib.sum(axis=1)
    return contrib


def feature_importance(ranked: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """
    Portfolio-level 'Top contributors' report: average weighted contribution of
    each group across the selected names, sorted high to low. Run every
    rebalance to track which factors are actually driving selection.
    """
    contrib = contributions(ranked, weights)
    imp = contrib.drop(columns="total").mean().sort_values(ascending=False)
    imp.name = "avg_contribution"
    return imp


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import data_engine as de
    syms = [p.stem for p in fe.FEATURE_DIR.glob("*.parquet")]
    if not syms:
        print("No features cached. Run data_engine.py then feature_engine.py.")
    else:
        date = pd.Timestamp.today()
        cfg = StrategyConfig(top_n=5)
        ranked = rank_universe(syms, date, cfg)
        print(f"Strategy: {cfg.name}  |  weights: {ranked.attrs.get('weights')}")
        print(f"Risk-filter drops: {ranked.attrs.get('drops')}\n")
        print(ranked.round(3))
        print("\nNOTE: shortlist to research, NOT buy signals.")
