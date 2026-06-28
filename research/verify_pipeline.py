"""
Pipeline verification — run on REAL data, on your machine
=========================================================
Purpose: before researching novel ideas, confirm (a) the data is trustworthy and
(b) the engine reproduces a well-documented strategy's *qualitative* behavior.

This script does NOT change any engine. It runs the canonical 12-1 momentum
strategy (top 20, monthly, equal weight) through the existing runner and prints
the result alongside the baselines, with literature-expectation guidance.

Run (on a networked machine, after `pip install yfinance pandas numpy matplotlib pyarrow`):
    cd research
    python verify_pipeline.py            # fetches real data, builds, runs, reports

Prep you must do once (real universe + benchmark):
    import sys; sys.path.append("../engines")
    import data_engine as de
    nifty500 = [...]                      # from niftyindices.com CSV (Symbol column)
    de.save_universe_snapshot(nifty500, "2026-06")
    # the runner will fetch prices + benchmark for you on first run

What to look for (sanity, not precision):
  - Momentum should show a positive long-run premium over the sample, but with
    deep drawdowns (momentum crashes). A flat/negative *and* low-vol result is a
    red flag to audit data.
  - It should beat the random-portfolio median; compare honestly to buy & hold.
  - Sharpe far above ~1.5 on a long sample is suspicious (look-ahead/data error),
    not a triumph.
  - Validation may FAIL — that's fine; this is a correctness check, not a hunt
    for a winner.

Research scaffolding, not investment advice.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "engines"))

from runner import RunConfig, run            # noqa: E402
from strategies import CLASSIC_MOMENTUM      # noqa: E402

# Adjust these for your environment
UNIVERSE_DATE = "2026-06"      # a snapshot you've saved via save_universe_snapshot
BENCHMARK = "^CRSLDX"          # Nifty 500 TRI (or "^NSEI" for Nifty 50)
START = "2012-01-01"           # ~12+ years if data allows


def main(fetch: bool = True):
    cfg = RunConfig(
        name="verify_classic_momentum",
        universe_date=UNIVERSE_DATE,
        benchmark=BENCHMARK,
        start=START,
        strat=CLASSIC_MOMENTUM,
        fetch=fetch,            # set False to reuse an existing cache
        do_baselines=True,
        do_validation=True,
        n_random=200,
    )
    out = run(cfg)

    perf = out["metrics"]["Performance"]
    print("\n================ VERIFICATION: classic 12-1 momentum ================")
    print(f"data: {out['data_kind']}   symbols: {out['n_symbols']}   run: {out['run_id']}")
    print(f"CAGR {perf['CAGR']}  Sharpe {perf['Sharpe']}  MaxDD {perf['Max Drawdown']}")
    if out.get("benchmark"):
        bm = out["benchmark"]
        print(f"vs benchmark: alpha {bm.get('Alpha (ann)')}  beta {bm.get('Beta')}  "
              f"IR {bm.get('Information Ratio')}")
    print(f"validation PASS: {out['passed']}")
    print(f"report: {out['report']}")
    print(f"all experiments: {out['index']}")
    print("\nInterpretation: confirm the SHAPE matches momentum's known profile")
    print("(positive premium, deep crash drawdowns). Audit data if wildly off.")
    print("=====================================================================")


if __name__ == "__main__":
    # Default tries to fetch real data. On a restricted machine, prepopulate the
    # cache and call main(fetch=False).
    main(fetch=True)
