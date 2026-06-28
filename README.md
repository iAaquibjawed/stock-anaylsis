# Quant Engine — Indian Equity Research Scaffold

A modular, look-ahead-safe research pipeline for screening NSE stocks.
**Research scaffolding, not investment advice. Outputs are candidates to research, never buy signals.**

**Status: V1.0 — complete research workflow.** All engines are built and tested
(on synthetic data, which validates correctness — not profitability). The next
step is empirical, not architectural: run it on real, survivorship-aware market
data and see whether any strategy clears the validation gate. Build ~10% more
infrastructure, spend ~90% on research from here.

## Structure

```
quant_engine/
├── engines/
│   ├── data_engine.py      # ✅ Engine 1: fetch / clean / validate / cache prices
│   ├── feature_engine.py   # ✅ Engine 2: causal feature matrix + metadata registry
│   ├── scoring_engine.py   # ✅ Engine 3: generic ranking + factor attribution
│   ├── execution_engine.py # ✅ Execution Simulator: fills, costs, P&L (reusable)
│   ├── backtest_engine.py  # ✅ Engine 4: rebalance loop + metrics + snapshots + journal
│   ├── validation_engine.py# ✅ Engine 5: the gatekeeper (PASS/FAIL)
│   ├── benchmark.py        # ✅ index-relative metrics (alpha/beta/TE/IR) + RS feature
│   ├── report.py           # ✅ self-contained HTML research report per experiment
│   ├── runner.py           # ✅ end-to-end orchestrator (universe -> report)
│   ├── experiments.py      # ✅ experiment ledger + index.html (compare runs)
│   └── baselines.py        # ✅ naive-strategy comparison (beat these first)
├── research/               # configurations & scripts (not engine code)
│   ├── strategies.py       # preset StrategyConfigs incl. CLASSIC_MOMENTUM
│   └── verify_pipeline.py  # run the canonical strategy on real data to verify
├── reports/                # generated reports, experiments.csv/jsonl, index.html
├── cache/                  # auto-created: prices, features, actions, meta (Parquet/JSON)
└── universe/               # auto-created: point-in-time constituent snapshots
```

Build order (each layer depends on the one above). Per review, **Backtest +
Validation come before Portfolio** — an attractive backtest must survive
validation before any portfolio is constructed. The Execution Simulator is split
out so the *same* fill logic drives both backtesting and future paper trading:
`Data → Feature → Scoring → Execution → Backtest → Validation → Portfolio → Live`

## Setup

```bash
pip install yfinance pandas numpy pyarrow
```

## Run

```bash
cd engines
python data_engine.py        # builds price cache for the universe
python feature_engine.py     # builds feature matrix from cached prices
python scoring_engine.py     # ranks the universe with the default strategy
python backtest_engine.py    # simulates the strategy through history with costs
python validation_engine.py  # runs the gatekeeper: PASS/FAIL verdict
```

### Real-data run (the whole pipeline in one call)

On a machine with internet, this fetches real NSE data, builds features, backtests,
benchmarks, validates, and writes an HTML report:

```python
from runner import RunConfig, run
from scoring_engine import StrategyConfig

cfg = RunConfig(
    name="momentum_nifty500",
    universe_csv="../universe/2026-06.csv",   # NSE constituents you saved
    benchmark="^CRSLDX",                       # Nifty 500 TR (or ^NSEI for Nifty 50)
    start="2015-01-01",
    strat=StrategyConfig(top_n=15),
)
summary = run(cfg)
print(summary["report"], summary["passed"])
```

`fetch=False` reuses an existing cache (offline/restricted environments). The
report is labeled **real** vs **synthetic** so numbers are never misread. A sample
synthetic report is in `reports/sample_report_synthetic.html`.

By default the Data Engine uses a 10-stock large-cap sample so it runs out of the box.
For the full Nifty 500: download NSE's constituents CSV from niftyindices.com, then
`save_universe_snapshot(symbols, "2026-06")` and `load_universe("2026-06")`.

## What each engine guarantees

**Data Engine (v2)** — incremental fetch, retry/backoff, corporate actions, data
validation, JSON metadata, parallel downloads, and a point-in-time universe to
avoid survivorship bias. `as_of(symbol, date)` is the anti-look-ahead gateway.

**Feature Engine (v2)** — momentum, trend (EMA, golden cross, 52w-high distance,
ADX, Donchian breakout), RSI, MACD, volatility (annualized vol, ATR%, gap%),
liquidity (avg daily value traded, volume spikes), and a signed consecutive-day
run. Ships a **metadata registry** (`FEATURE_META`) declaring each feature's
group, direction (higher/lower better), and normalization — so the Scoring
Engine stays generic. `FEATURE_SCHEMA_VERSION` versions the set for reproducible
backtests. Every feature is **causal**: the value on bar *t* uses only data known
at the close of *t*, verified by a truncation test (recomputing on a price series
cut at *T* gives byte-identical values, checked for ADX too).
`features_as_of(symbol, date)` and `cross_section(symbols, date)` are the gateways.

**Scoring Engine (Engine 3)** — generic and **config-driven via `StrategyConfig`**:
no hard-coded weights. Pipeline is risk-filter → cross-sectional normalize
(percentile rank or winsorized z-score) → direction-correct → group-weighted
score → rank. Reads semantics from `FEATURE_META`, so swapping the config swaps
the strategy (momentum / quality / mean-reversion) with no code changes. Risk
filters (liquidity, volatility, penny, uptrend) run *before* ranking so noisy
signals on untradeable names can't win. `rank_universe(symbols, date, cfg)`
returns the shortlist. **Factor attribution**: `contributions()` shows the
weighted per-group contribution to each stock's score (rows sum to the score),
and `feature_importance()` gives a "top contributors" report — so when a strategy
stops working you can see exactly which factor broke.

**Execution Simulator** (`execution_engine.py`) — a reusable, stateful broker
model. Given target weights and the day's open prices it models next-open fills,
commissions (bps), slippage (adverse fills), integer lot sizes, ADV-based
liquidity caps, and a no-leverage cash balance, while tracking realized P&L
(average-cost) and holding periods. Split out so the identical fill logic powers
both historical backtests and future paper trading — making paper results
directly comparable.

**Backtest Engine (Engine 4)** — the rebalance loop with one rule enforced
*structurally*, not by convention: **features/scores use the signal day's close;
orders fill on the next trading day's open**, so we never trade on the same close
that generated the signal. Valuation uses AdjClose; fills use an adjusted open
(`Open × AdjClose/Close`) to keep both on one split-consistent scale. Reports
from day one: CAGR, annualized return/vol, Sharpe, Sortino, Calmar, max drawdown,
period win rate, profit factor; turnover, avg holding period, trade counts,
commission and slippage paid, cash drag; target positions, realized win rate,
end concentration (HHI), and a factor-importance summary. Beta vs Nifty is a
declared hook — it needs an index series, which isn't wired yet (not faked).
It also emits **per-rebalance portfolio snapshots** (cash, equity, exposure,
HHI, turnover, drawdown) and a **trade journal** — one permanent record per
realized round trip (entry/exit dates and prices, return, holding days) so you
can later mine which conditions win or lose.

**Validation Engine (Engine 5) — the gatekeeper.** A strategy only earns a
portfolio if its edge survives attack. Eight checks across three families, each
with a PASS/FAIL verdict; the strategy passes only if all *critical* checks pass:

- Statistical: `out_of_sample` (edge persists on held-out data), `rolling_windows`
  (positive across most overlapping windows), `bootstrap_ci` (block-bootstrap 95%
  Sharpe CI with lower bound > 0), `monte_carlo` (actual drawdown not in the worst
  tail of reshuffled paths).
- Robustness: `cost_stress` (+25% commission / +50% slippage still profitable),
  `weight_perturb` (jittered weights stay positive), `frequency` (edge survives a
  different rebalance cadence).
- Risk: `risk_limits` (drawdown, turnover, concentration, min-Sharpe bounds).

Every run records **provenance** — a config hash, feature schema version, RNG
seed, and timestamp — so any verdict is reproducible months later. `validate(...)`
returns a `ValidationReport`; `print_report()` shows the scorecard.

**Benchmark module** (`benchmark.py`) — index-relative metrics (CAPM alpha, beta,
tracking error, information ratio, excess CAGR) computed from a *cached* index
series, plus a causal relative-strength feature. Nothing is faked: these only
appear when a real index (e.g. `^NSEI`, `^CRSLDX`) is actually cached.

**Report generator** (`report.py`) — one self-contained HTML artifact per
experiment: provenance, the validation scorecard, full metrics, benchmark block,
equity and drawdown charts, factor attribution, and a trade-journal summary. A
prominent banner states whether the run used **real** or **synthetic** data.

**Runner** (`runner.py`) — the end-to-end orchestrator: resolve universe →
fetch/cache prices → build features → backtest → benchmark → validate → report →
register, all from one `RunConfig`. This is the bridge from "tested on synthetic"
to "run on a genuine historical universe".

**Experiment Manager** (`experiments.py`) — a lightweight ledger (two flat files,
no heavy DB) so runs are comparable instead of isolated. The runner auto-registers
every run to `reports/experiments.csv` (the comparison table) and
`reports/experiments.jsonl` (full record incl. config hash), and rebuilds
`reports/index.html` — a table of all experiments with Sharpe, CAGR, max DD,
turnover, validation verdict, and a link to each report. `compare(sort_by=...)`
ranks them. Synthetic rows are tagged distinctly so only **real** rows read as
research results. Open `reports/index.html` to browse all runs.

**Baselines** (`baselines.py`) — a strategy that beats the index but loses to a
coin-flip portfolio hasn't proven anything. This compares the strategy, *after
costs and on the same next-open timing*, against: buy & hold index, buy & hold
equal-weight, equal-weight rebalanced, top-N 12-month momentum, and a cloud of
random portfolios (many sims) — reporting the strategy's percentile vs random.
The runner runs these by default (`do_baselines=True`) and the report shows the
table. **If the strategy can't beat these, the strategy needs work — not the
code.** (In the included sample, plain momentum beats the configured strategy —
exactly the kind of useful finding this surfaces.)

## Honest limitations (read these)

1. **`as_of()` is PIT-correct for PRICES only.** yfinance serves *restated*
   fundamentals, so do not backtest P/E- or earnings-based rules as if those
   numbers were known at the time. Fundamentals are deliberately excluded from
   the feature matrix for this reason.
2. **Survivorship fix works going forward.** `save_universe_snapshot` builds
   history from when you start capturing it; true 2018 constituents need an
   archived list.
3. **No screening loop guarantees profit.** This finds statistically interesting
   candidates to research. The edge (if any) is statistical and only shows up
   across many trades after realistic costs — proven by the Backtesting and
   Validation engines, which are the next pieces to build.

## Deliberately deferred (with rationale)

These were reviewed and intentionally not built yet — building them now would add
complexity ahead of need:

- **Event-driven engine + Order objects.** The current engine is rebalance-driven,
  which is sufficient to answer the only question that matters right now (does the
  edge survive validation?). An event loop pays off for stop-losses, intraday
  exits, partial fills, and live execution — worth the rewrite only once a
  strategy has actually passed the gate and is heading toward live.
- **Research database.** A thin persistence layer for experiments. The provenance
  block (config hash + versions + seed) is already emitted; wiring it to a store
  is a small step best taken once validation is producing experiments worth
  keeping.
- **Index-dependent features.** Regime (index vs 200DMA, India VIX, breadth),
  relative strength, and beta-vs-Nifty all need an index series that isn't wired.
  Fundamentals need a point-in-time source. Hooks are ready; data isn't.

## First real-data run — a phased runbook

Don't jump to "is it accurate?". Answer these in order; treat Phases 1–2 as
correctness, Phase 3 as measurement, Phase 4 as research.

**Phase 1 — Verify the data.** Download the full Nifty 500 list, save it as a
universe snapshot, cache 10+ years of prices and the benchmark, and check for
gaps/bad splits.

```python
import data_engine as de
nifty500 = [...]                              # from niftyindices.com CSV
de.save_universe_snapshot(nifty500, "2026-06")
de.build_cache(nifty500 + ["^CRSLDX"], start="2014-01-01")
for s in nifty500[:10]:
    print(s, de.get_meta(s)["warnings"])     # spot-check data-quality flags
```

**Phase 2 — Verify the engine (correctness, not profit).** Run the whole pipeline
once and confirm features build, rankings look sane, trades land on the next open,
and the report/validation complete.

```python
from runner import RunConfig, run
from scoring_engine import StrategyConfig
out = run(RunConfig(name="momentum_smoke", universe_date="2026-06",
                    benchmark="^CRSLDX", start="2015-01-01",
                    strat=StrategyConfig(top_n=15)))
print(out["report"])
```

**Phase 3 — Measure.** Read the report: CAGR, Sharpe, max DD, win rate, turnover,
alpha/beta/IR vs benchmark, the validation verdict — and the baseline table.
Failing here is normal and informative.

**Phase 3.5 — Verify against a known result.** Before researching novel ideas,
run the canonical strategy and check its *shape* against the literature:

```bash
cd research && python verify_pipeline.py     # classic 12-1 momentum, top 20, monthly
```

Classic 12‑1 momentum (`feature: mom_12_1`, feature schema v3) lives in its own
`classic_momentum` group with **zero weight in the default config**, so it's
opt-in and changes no existing strategy. Expect a positive long-run premium with
deep "momentum crash" drawdowns. Results wildly different from that profile mean
*audit the data/implementation first* — don't start researching on a broken base.
You won't match published returns exactly (different universe, costs, impl), and
that's fine; you're checking qualitative behavior, not precision.

**Phase 4 — Research (90% of the work).** Only now ask which factors help or hurt,
whether momentum/quality/mean-reversion work, and whether weights should change —
letting the evidence drive each iteration. If a strategy fails, understand *why*
before tweaking. If it passes, re-run across different periods and regimes to see
if the edge persists. Every run is logged to `reports/index.html` for comparison.

## The one thing that still matters most: real data

The pipeline is complete and the runner can drive it end-to-end, but **all
metrics shown so far come from synthetic data** — they prove the framework is
correct (timing, statistics, determinism, gate logic), not that any strategy is
profitable. To get real research results you need to supply real data:

1. **Universe history** — save monthly NSE constituent snapshots
   (`data_engine.save_universe_snapshot`). Today's list ≠ the historical list;
   without archived constituents, older backtests still carry survivorship bias.
2. **Benchmark series** — cache `^NSEI` / `^CRSLDX` so alpha/beta/IR populate.
3. **Run it** — `runner.run(RunConfig(..., fetch=True))` on a networked machine.
4. **Fundamentals stay excluded** until a point-in-time source is available
   (yfinance restates them; including them would leak the future).

Only a strategy that clears the gate **on real data** should advance to Portfolio
construction, then paper trading. Automated live trading stays off the table
until paper results confirm the edge.
