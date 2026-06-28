"""
Real-data Runner — end-to-end orchestrator
==========================================
One command takes a strategy from raw data to a validated report:

    universe -> fetch/cache prices -> features -> backtest -> benchmark
             -> validation -> HTML report

This is the piece that turns the platform from "tested on synthetic data" into
"run on a genuine historical universe". On a machine with internet it fetches
real NSE data via yfinance; in a restricted environment pass fetch=False and
point it at an already-cached universe.

Usage (on your machine):
    from runner import RunConfig, run
    cfg = RunConfig(
        name="momentum_nifty500",
        universe_csv="../universe/2026-06.csv",   # NSE constituents you saved
        benchmark="^CRSLDX",                       # Nifty 500 TR index (or ^NSEI)
        start="2015-01-01",
    )
    summary = run(cfg)
    print(summary["report"])   # path to the HTML report

Honest scope: real fundamentals stay point-in-time-unavailable (excluded), and
universe history is only as deep as the snapshots you've saved. The runner labels
the report 'real' vs 'synthetic' so results are never misread.

Research scaffolding, not investment advice.

Dependencies:
    pip install yfinance pandas numpy matplotlib pyarrow
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import data_engine as de
import feature_engine as fe
import scoring_engine as se
import backtest_engine as bt
import validation_engine as ve
import benchmark as bm
import report as rp
import experiments as ex
from execution_engine import ExecConfig


@dataclass
class RunConfig:
    name: str = "experiment"
    # universe: provide ONE of universe_csv / universe_date / symbols
    universe_csv: str | None = None        # path to a constituents CSV (Symbol col)
    universe_date: str | None = None       # use a saved snapshot effective on date
    symbols: list[str] | None = None       # explicit list
    benchmark: str | None = None           # benchmark ticker, e.g. "^CRSLDX"
    start: str = "2015-01-01"
    end: str | None = None
    strat: se.StrategyConfig = field(default_factory=se.StrategyConfig)
    exec_cfg: ExecConfig = field(default_factory=ExecConfig)
    vcfg: ve.ValidationConfig = field(default_factory=ve.ValidationConfig)
    starting_cash: float = 1_000_000.0
    fetch: bool = True                     # False -> use existing cache only
    do_validation: bool = True
    out_dir: str = "../reports"


def _resolve_symbols(cfg: RunConfig) -> list[str]:
    if cfg.symbols:
        return cfg.symbols
    if cfg.universe_csv:
        col = "Symbol"
        df = pd.read_csv(cfg.universe_csv)
        col = col if col in df.columns else df.columns[0]
        return df[col].astype(str).str.strip().tolist()
    if cfg.universe_date:
        return de.load_universe(cfg.universe_date)
    return de.load_universe()               # most recent snapshot or sample


def run(cfg: RunConfig) -> dict:
    symbols = _resolve_symbols(cfg)
    data_kind = "synthetic" if not cfg.fetch else "real"

    # 1) data
    if cfg.fetch:
        to_fetch = list(symbols) + ([cfg.benchmark] if cfg.benchmark else [])
        de.build_cache(to_fetch, start=cfg.start)

    # keep only symbols that actually have cached prices
    cached = {p.stem for p in de.CACHE_DIR.glob("*.parquet")}
    symbols = [s for s in symbols if s in cached]
    if not symbols:
        raise RuntimeError("No cached price data for the universe. "
                           "Run with fetch=True on a networked machine, or "
                           "pre-populate the cache.")

    # 2) features
    fe.clear_cache()
    fe.build_all_features(symbols)

    # 3) backtest
    res = bt.run_backtest(symbols, cfg.start, cfg.end, cfg.strat,
                          cfg.exec_cfg, cfg.starting_cash)

    # 4) benchmark (only if cached — never faked)
    bench_metrics = None
    if cfg.benchmark and cfg.benchmark in cached:
        try:
            bench_close = de.get(cfg.benchmark)["AdjClose"]
            bench_metrics = bm.benchmark_metrics(res.equity, bench_close)
        except Exception as e:  # noqa: BLE001
            bench_metrics = {"error": f"{type(e).__name__}: {e}"}

    # 5) validation
    val = None
    if cfg.do_validation:
        val = ve.validate(symbols, cfg.start, cfg.end, cfg.strat,
                          cfg.exec_cfg, cfg.vcfg)

    # 6) report
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"{cfg.name}_{stamp}.html"
    provenance = (val.provenance if val else
                  {"timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                   "feature_schema_version": fe.FEATURE_SCHEMA_VERSION,
                   "n_symbols": len(symbols)})
    rp.generate_report(res, str(report_path), title=cfg.name, validation=val,
                       bench_metrics=bench_metrics, data_kind=data_kind,
                       provenance=provenance)

    # 7) register in the experiment ledger (auto-rebuilds reports/index.html)
    universe_label = (cfg.universe_csv or cfg.universe_date or "explicit") if not cfg.symbols else "explicit"
    ledger_row = ex.register(
        strategy=cfg.name,
        metrics=res.metrics,
        report_path=str(report_path),
        universe=str(universe_label),
        n_symbols=len(symbols),
        period=f"{cfg.start}..{cfg.end or 'today'}",
        data_kind=data_kind,
        validation=(val.passed if val else None),
        config_hash=provenance.get("config_hash", ""),
        full_record={"strat": cfg.strat.__dict__, "benchmark": bench_metrics},
    )

    return {
        "report": str(report_path),
        "index": str(ex.INDEX_PATH),
        "run_id": ledger_row["run_id"],
        "passed": (val.passed if val else None),
        "metrics": res.metrics,
        "benchmark": bench_metrics,
        "n_symbols": len(symbols),
        "data_kind": data_kind,
    }


if __name__ == "__main__":
    # Default run uses whatever is cached (sample universe if nothing else).
    cfg = RunConfig(name="demo", fetch=True)
    out = run(cfg)
    print(f"\nReport: {out['report']}")
    print(f"Validation passed: {out['passed']}  |  data: {out['data_kind']}")
