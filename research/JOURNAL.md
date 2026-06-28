# Research Journal

The experiment **ledger** (`reports/experiments.csv`) records *what* happened —
metrics, validation verdict, config hash. This journal records *why*. One entry
per hypothesis. Over time these notes prevent rediscovering the same dead ends,
and they usually outlast the code in value.

For each idea capture: the **hypothesis**, **why you expected it to work**, the
**result**, **why** you think it passed/failed, and **what you learned**. Link the
`run_id` / `config_hash` from the ledger so the entry is reproducible.

The milestone that matters isn't "V1.1" — it's something like: *"Tested N
hypotheses on 15y of Nifty 500. M failed validation. K passed the gate. J beat
the benchmark and all baselines after costs, and are different enough to
diversify."* Track toward that, not toward commit count.

Rules of engagement (from V1 close):
- Don't add code because it seems useful — only when an experiment exposes a real
  limitation (e.g. a strategy needs sector-neutral ranking → then add it).
- Don't optimize until something already beats the baselines in its first form.
  Tuning before that fits noise.

---

## Template (copy per experiment)

### EXP-XXX — <short name>
- **Date:**
- **run_id / config_hash:** (from reports/experiments.csv)
- **Hypothesis:**
- **Why I expected it to work:** (economic / behavioral / structural rationale)
- **Setup:** universe, period, top_n, rebalance, costs, key config
- **Result:** backtest CAGR / Sharpe / MaxDD · validation PASS/FAIL · vs baselines
- **Why it passed/failed:** (read the factor attribution, baselines, drawdowns)
- **What I learned / next action:**

---

## Example entry (fill in after the first real run)

### EXP-000 — Platform verification: classic 12-1 momentum
- **Date:** <when you run it>
- **run_id / config_hash:** <from ledger>
- **Hypothesis:** Classic 12-1 momentum (top 20, monthly, equal weight) earns a
  positive long-run premium on Nifty 500.
- **Why I expected it to work:** Well-documented cross-sectional momentum premium;
  this run is a *verification* of the platform, not a novel idea.
- **Setup:** universe = Nifty 500 snapshot; ~12-15y; top_n=20; monthly; default costs.
- **Result:** <CAGR / Sharpe / MaxDD> · validation <PASS/FAIL> · vs baselines <...>
- **Why it passed/failed:** <does the SHAPE match literature — positive premium
  with deep momentum-crash drawdowns? If wildly off, audit data/implementation.>
- **What I learned / next action:** <trust the platform and start research, OR fix
  a data/implementation issue before proceeding.>
