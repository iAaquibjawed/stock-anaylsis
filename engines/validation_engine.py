"""
Validation Engine (Engine 5) — the gatekeeper
=============================================
A strategy does NOT earn the right to a portfolio just because its backtest looks
good. This engine attacks the strategy and only lets it through if the edge
survives. Every check returns a verdict; the strategy PASSES only if all
*critical* checks pass.

Checks
------
Statistical
  - out_of_sample      : split history; edge must persist on unseen data
  - rolling_windows    : Sharpe positive across most overlapping windows
  - bootstrap_ci       : block-bootstrap 95% CI on Sharpe; lower bound > 0
  - monte_carlo        : actual max drawdown not extreme vs resampled paths
Robustness
  - cost_stress        : +25% commission, +50% slippage -> still profitable
  - weight_perturb     : jitter group weights -> Sharpe stays positive
  - frequency          : different rebalance cadence -> edge survives
Risk limits
  - risk_limits        : drawdown / turnover / concentration within bounds

Reproducibility: every run records a config hash, the feature/score schema
versions, the RNG seed, and a timestamp, so any verdict can be reproduced.

This finds reasons to REJECT. Passing is necessary, not sufficient — paper
trading is still the next gate. Research scaffolding, not investment advice.

Dependencies:
    pip install pandas numpy
"""
from __future__ import annotations

import json
import hashlib
import datetime as dt
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

import feature_engine as fe
import scoring_engine as se
import backtest_engine as bt
from execution_engine import ExecConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ValidationConfig:
    oos_fraction: float = 0.30        # last 30% of history held out
    n_rolling: int = 4               # overlapping windows
    rolling_min_positive: float = 0.60
    n_bootstrap: int = 1000
    block: int = 20                  # block size for block bootstrap (~1 month)
    n_mc: int = 1000
    mc_dd_percentile: float = 0.05   # actual DD must beat the 5th pct of sims
    cost_commission_mult: float = 1.25
    cost_slippage_mult: float = 1.50
    weight_jitter: float = 0.25      # +/-25% relative jitter
    n_perturb: int = 5
    freqs: tuple[str, ...] = ("Q",)  # alt cadence(s) vs the baseline
    # Risk limits
    max_drawdown_limit: float = 0.35
    max_turnover_ann: float = 25.0
    max_concentration_hhi: float = 0.50
    min_sharpe: float = 0.30
    seed: int = 42


@dataclass
class Check:
    name: str
    passed: bool
    critical: bool
    detail: str
    value: float | None = None
    threshold: float | None = None


@dataclass
class ValidationReport:
    strategy: str
    passed: bool
    checks: list[Check] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.checks])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sharpe_from_equity(eq: pd.Series) -> float:
    r = eq.pct_change().dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd else np.nan


def _config_hash(symbols, start, end, strat, exec_cfg, vcfg) -> str:
    blob = json.dumps({
        "symbols": sorted(symbols), "start": start, "end": str(end),
        "strat": asdict(strat), "exec": asdict(exec_cfg or ExecConfig()),
        "vcfg": asdict(vcfg),
    }, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _trading_dates(symbols, start, end):
    panel = bt.build_panel(symbols)
    return panel["close"].loc[start:end].index


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _check_out_of_sample(symbols, start, end, strat, exec_cfg, vcfg) -> Check:
    days = _trading_dates(symbols, start, end)
    split_i = int(len(days) * (1 - vcfg.oos_fraction))
    split = days[split_i]
    is_res = bt.run_backtest(symbols, start, str(split.date()), strat, exec_cfg)
    oos_res = bt.run_backtest(symbols, str(split.date()), end, strat, exec_cfg)
    is_s = _sharpe_from_equity(is_res.equity)
    oos_s = _sharpe_from_equity(oos_res.equity)
    # OOS must be positive and retain at least half the in-sample Sharpe
    ok = (oos_s > 0) and (oos_s >= 0.5 * is_s if is_s and is_s > 0 else oos_s > 0)
    return Check("out_of_sample", bool(ok), True,
                 f"IS Sharpe {is_s:.2f} -> OOS Sharpe {oos_s:.2f} (split {split.date()})",
                 round(oos_s, 3), round(0.5 * is_s, 3) if is_s and is_s > 0 else 0.0)


def _check_rolling(symbols, start, end, strat, exec_cfg, vcfg) -> Check:
    days = _trading_dates(symbols, start, end)
    n = len(days)
    win = n // 2
    starts = np.linspace(0, n - win - 1, vcfg.n_rolling).astype(int)
    sharpes = []
    for s0 in starts:
        d0, d1 = days[s0], days[min(s0 + win, n - 1)]
        res = bt.run_backtest(symbols, str(d0.date()), str(d1.date()), strat, exec_cfg)
        sharpes.append(_sharpe_from_equity(res.equity))
    sharpes = [x for x in sharpes if x == x]
    frac_pos = np.mean([x > 0 for x in sharpes]) if sharpes else 0.0
    ok = frac_pos >= vcfg.rolling_min_positive
    return Check("rolling_windows", bool(ok), True,
                 f"{frac_pos:.0%} of {len(sharpes)} windows positive "
                 f"(Sharpes {[round(x,2) for x in sharpes]})",
                 round(frac_pos, 3), vcfg.rolling_min_positive)


def _block_bootstrap_sharpe(returns: np.ndarray, n_boot, block, rng) -> np.ndarray:
    n = len(returns)
    if n < block * 2:
        block = max(1, n // 4)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([returns[i:i + block] for i in idx])[:n]
        sd = sample.std()
        out[b] = sample.mean() / sd * np.sqrt(252) if sd else 0.0
    return out


def _check_bootstrap(base_res, vcfg, rng) -> Check:
    r = base_res.equity.pct_change().dropna().to_numpy()
    if len(r) < 30:
        return Check("bootstrap_ci", False, True, "too few returns", None, 0.0)
    dist = _block_bootstrap_sharpe(r, vcfg.n_bootstrap, vcfg.block, rng)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    ok = lo > 0
    return Check("bootstrap_ci", bool(ok), True,
                 f"Sharpe 95% CI [{lo:.2f}, {hi:.2f}] (lower bound must be > 0)",
                 round(float(lo), 3), 0.0)


def _check_monte_carlo(base_res, vcfg, rng) -> Check:
    r = base_res.equity.pct_change().dropna().to_numpy()
    if len(r) < 30:
        return Check("monte_carlo", False, False, "too few returns", None, None)
    actual_dd = _max_dd_from_returns(r)
    sims = np.empty(vcfg.n_mc)
    for i in range(vcfg.n_mc):
        shuffled = rng.permutation(r)
        sims[i] = _max_dd_from_returns(shuffled)
    # actual DD should not be in the worst tail of reshuffled paths
    pctile = float((sims < actual_dd).mean())  # fraction of sims worse (more neg)
    ok = pctile >= vcfg.mc_dd_percentile
    return Check("monte_carlo", bool(ok), False,
                 f"actual maxDD {actual_dd:.1%}; {pctile:.0%} of shuffled paths worse",
                 round(pctile, 3), vcfg.mc_dd_percentile)


def _max_dd_from_returns(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def _check_cost_stress(symbols, start, end, strat, exec_cfg, vcfg) -> Check:
    base = exec_cfg or ExecConfig()
    stressed = ExecConfig(
        commission_bps=base.commission_bps * vcfg.cost_commission_mult,
        slippage_bps=base.slippage_bps * vcfg.cost_slippage_mult,
        lot_size=base.lot_size, max_participation=base.max_participation,
        allow_fractional=base.allow_fractional,
    )
    res = bt.run_backtest(symbols, start, end, strat, stressed)
    s = _sharpe_from_equity(res.equity)
    ok = s > 0
    return Check("cost_stress", bool(ok), True,
                 f"+{(vcfg.cost_commission_mult-1):.0%} comm / +{(vcfg.cost_slippage_mult-1):.0%} "
                 f"slip -> Sharpe {s:.2f}", round(s, 3), 0.0)


def _check_weight_perturb(symbols, start, end, strat, exec_cfg, vcfg, rng) -> Check:
    sharpes = []
    base_w = strat.group_weights
    for _ in range(vcfg.n_perturb):
        jitter = {g: max(0.0, w * (1 + rng.uniform(-vcfg.weight_jitter, vcfg.weight_jitter)))
                  for g, w in base_w.items()}
        s2 = se.StrategyConfig(name=strat.name, group_weights=jitter,
                               normalize=strat.normalize, winsor=strat.winsor,
                               top_n=strat.top_n, min_adv_20=strat.min_adv_20,
                               max_vol_20d=strat.max_vol_20d, min_price=strat.min_price,
                               require_uptrend=strat.require_uptrend)
        res = bt.run_backtest(symbols, start, end, s2, exec_cfg)
        sharpes.append(_sharpe_from_equity(res.equity))
    sharpes = [x for x in sharpes if x == x]
    frac_pos = np.mean([x > 0 for x in sharpes]) if sharpes else 0.0
    ok = frac_pos >= 0.8
    return Check("weight_perturb", bool(ok), False,
                 f"{frac_pos:.0%} of {len(sharpes)} perturbations positive",
                 round(frac_pos, 3), 0.8)


def _check_frequency(symbols, start, end, strat, exec_cfg, vcfg) -> Check:
    results = {}
    for f in vcfg.freqs:
        res = bt.run_backtest(symbols, start, end, strat, exec_cfg, freq=f)
        results[f] = _sharpe_from_equity(res.equity)
    ok = all((s > 0) for s in results.values() if s == s)
    return Check("frequency", bool(ok), False,
                 f"alt-cadence Sharpes { {k: round(v,2) for k,v in results.items()} }",
                 None, 0.0)


def _check_risk_limits(base_res, vcfg) -> Check:
    m = base_res.metrics
    mdd = abs(m["Performance"]["Max Drawdown"])
    turn = m["Trading"]["Turnover (ann)"] or 0.0
    hhi = m["Portfolio"]["End Concentration (HHI)"] or 0.0
    sharpe = m["Performance"]["Sharpe"] or 0.0
    fails = []
    if mdd > vcfg.max_drawdown_limit: fails.append(f"maxDD {mdd:.0%}>{vcfg.max_drawdown_limit:.0%}")
    if turn > vcfg.max_turnover_ann: fails.append(f"turnover {turn:.1f}>{vcfg.max_turnover_ann}")
    if hhi > vcfg.max_concentration_hhi: fails.append(f"HHI {hhi:.2f}>{vcfg.max_concentration_hhi}")
    if sharpe < vcfg.min_sharpe: fails.append(f"Sharpe {sharpe:.2f}<{vcfg.min_sharpe}")
    ok = not fails
    return Check("risk_limits", bool(ok), True,
                 "within limits" if ok else "; ".join(fails), None, None)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def validate(
    symbols: list[str],
    start: str,
    end: str | None = None,
    strat: se.StrategyConfig | None = None,
    exec_cfg: ExecConfig | None = None,
    vcfg: ValidationConfig | None = None,
) -> ValidationReport:
    strat = strat or se.StrategyConfig()
    vcfg = vcfg or ValidationConfig()
    rng = np.random.default_rng(vcfg.seed)

    base = bt.run_backtest(symbols, start, end, strat, exec_cfg)

    checks: list[Check] = []
    # statistical
    checks.append(_check_out_of_sample(symbols, start, end, strat, exec_cfg, vcfg))
    checks.append(_check_rolling(symbols, start, end, strat, exec_cfg, vcfg))
    checks.append(_check_bootstrap(base, vcfg, rng))
    checks.append(_check_monte_carlo(base, vcfg, rng))
    # robustness
    checks.append(_check_cost_stress(symbols, start, end, strat, exec_cfg, vcfg))
    checks.append(_check_weight_perturb(symbols, start, end, strat, exec_cfg, vcfg, rng))
    checks.append(_check_frequency(symbols, start, end, strat, exec_cfg, vcfg))
    # risk
    checks.append(_check_risk_limits(base, vcfg))

    passed = all(c.passed for c in checks if c.critical)
    provenance = {
        "config_hash": _config_hash(symbols, start, end, strat, exec_cfg, vcfg),
        "feature_schema_version": fe.FEATURE_SCHEMA_VERSION,
        "seed": vcfg.seed,
        "n_symbols": len(symbols),
        "period": f"{start}..{end or 'today'}",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    return ValidationReport(strat.name, passed, checks, provenance)


def print_report(rep: ValidationReport) -> None:
    print(f"\nVALIDATION — strategy '{rep.strategy}'")
    print(f"provenance: {rep.provenance}\n")
    for c in rep.checks:
        mark = "PASS" if c.passed else "FAIL"
        crit = "*" if c.critical else " "
        print(f"  [{mark}]{crit} {c.name:<16} {c.detail}")
    print(f"\n  (* = critical)\n  VERDICT: {'PASS — proceed to paper trading' if rep.passed else 'REJECT'}")
    print("\nNOTE: passing is necessary, not sufficient. Paper trade before real capital.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    syms = [p.stem for p in fe.FEATURE_DIR.glob("*.parquet")]
    if not syms:
        print("No features cached. Run data_engine.py then feature_engine.py.")
    else:
        rep = validate(syms, start="2019-01-01", strat=se.StrategyConfig(top_n=5))
        print_report(rep)
