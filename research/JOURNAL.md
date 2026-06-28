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

Guiding principle:
> Don't ask "Can I make this strategy pass?" Ask "What evidence would convince me
> to abandon it?" Assume no edge until the data proves otherwise. Let the data
> change your mind, not your intuition.

Rules of engagement (from V1 close):
- Fill the four PRE-RUN fields *before* you click Run, and don't edit them after.
  When the report returns, compare it to your prediction instead of rewriting the
  story.
- Don't add code because it seems useful — only when an experiment exposes a real
  limitation (e.g. a strategy needs sector-neutral ranking → then add it).
- Don't optimize until something already beats the baselines in its first form.
  Tuning before that fits noise.
- A rejected idea is a successful experiment. The goal is a *graveyard* of
  well-documented dead ends + a few survivors — e.g. 300 run / 280 rejected /
  18 inconclusive / 2 surviving is a productive year.

---

## Template (copy per experiment)

### EXP-XXX — <short name>

**PRE-RUN (write before running; do not edit afterward):**
- **Date:**
- **Hypothesis:** What do I believe?
- **Mechanism:** Why should this create an edge? (economic / behavioral / structural)
- **Falsification criterion:** What result would make me conclude this doesn't work?
- **Expected outcome:** My numeric prediction before seeing any results.
- **Setup:** universe, period, top_n, rebalance, costs, key config

**POST-RUN:**
- **run_id / config_hash:** (from reports/experiments.csv)
- **Result:** backtest CAGR / Sharpe / MaxDD · validation PASS/FAIL · vs baselines
- **Prediction vs reality:** how did it compare to my Expected outcome?
- **Verdict:** did it hit the falsification criterion? KEEP / REJECT / INCONCLUSIVE
- **Why it passed/failed:** (read factor attribution, baselines, drawdowns)
- **What I learned / next action:**

---

## Example entry (fill in after the first real run)

### EXP-000 — Platform verification: classic 12-1 momentum

**PRE-RUN:**
- **Date:** <when you run it>
- **Hypothesis:** Classic 12-1 momentum (top 20, monthly, equal weight) earns a
  positive long-run premium on Nifty 500.
- **Mechanism:** Well-documented cross-sectional momentum premium (under-reaction /
  persistence). This run *verifies the platform*, not a novel idea.
- **Falsification criterion:** Flat-or-negative return with shallow drawdowns, or a
  Sharpe so high (>~2 on 12-15y) it implies look-ahead — either means audit data/impl.
- **Expected outcome:** Positive premium with deep "momentum-crash" drawdowns;
  beats random + buy&hold; may or may not pass the strict validation gate.
- **Setup:** universe = Nifty 500 snapshot; ~12-15y; top_n=20; monthly; default costs.

**POST-RUN:**
- **run_id / config_hash:** <from ledger>
- **Result:** <CAGR / Sharpe / MaxDD> · validation <PASS/FAIL> · vs baselines <...>
- **Prediction vs reality:** <did the shape match what I wrote above?>
- **Verdict:** <KEEP / REJECT / INCONCLUSIVE>
- **Why it passed/failed:** <does the SHAPE match literature? If wildly off, audit.>
- **What I learned / next action:** <trust the platform and start research, OR fix
  a data/implementation issue first.>
