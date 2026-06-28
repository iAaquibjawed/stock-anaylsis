"""
Research Report generator
=========================
Produces ONE self-contained HTML report per experiment, so every backtest is a
permanent, comparable artifact. Charts are rendered with matplotlib and embedded
as base64 PNGs — no external files, the .html opens anywhere.

Sections:
  - header + provenance (config hash, versions, seed, timestamp)
  - validation verdict scorecard (if a ValidationReport is supplied)
  - performance / trading / portfolio metrics
  - benchmark metrics (if supplied)
  - equity curve, drawdown curve, factor attribution charts
  - trade journal summary

A REPRODUCIBILITY + SCOPE banner states plainly whether the run used synthetic or
real data, because metrics only mean something on real history.

Research scaffolding, not investment advice.

Dependencies:
    pip install pandas numpy matplotlib
"""
from __future__ import annotations

import base64
import io
import datetime as dt
import html as _html

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64: str, alt: str) -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}"/>'


def _equity_chart(eq: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(eq.index, eq.values, color="#1f6feb", lw=1.4)
    ax.set_title("Equity curve"); ax.grid(alpha=0.25)
    ax.set_ylabel("Portfolio value")
    return _img(_fig_to_b64(fig), "equity curve")


def _drawdown_chart(eq: pd.Series) -> str:
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.fill_between(dd.index, dd.values, 0, color="#d1242f", alpha=0.5)
    ax.set_title("Drawdown"); ax.grid(alpha=0.25)
    ax.set_ylabel("Drawdown")
    return _img(_fig_to_b64(fig), "drawdown")


def _attribution_chart(importance: pd.Series) -> str:
    if importance is None or importance.empty:
        return "<p><em>No factor attribution available.</em></p>"
    fig, ax = plt.subplots(figsize=(6, 3))
    imp = importance.sort_values()
    ax.barh(imp.index, imp.values, color="#2da44e")
    ax.set_title("Average factor attribution"); ax.grid(alpha=0.25, axis="x")
    return _img(_fig_to_b64(fig), "factor attribution")


def _metrics_table(metrics: dict) -> str:
    parts = []
    for section, vals in metrics.items():
        if section.startswith("_"):
            continue
        rows = "".join(
            f"<tr><td>{_html.escape(str(k))}</td><td>{_html.escape(str(v))}</td></tr>"
            for k, v in vals.items()
        )
        parts.append(f"<h3>{_html.escape(section)}</h3><table>{rows}</table>")
    return "".join(parts)


def _dict_table(title: str, d: dict) -> str:
    rows = "".join(
        f"<tr><td>{_html.escape(str(k))}</td><td>{_html.escape(str(v))}</td></tr>"
        for k, v in d.items()
    )
    return f"<h3>{_html.escape(title)}</h3><table>{rows}</table>"


def _validation_table(val) -> str:
    if val is None:
        return "<p><em>No validation run for this experiment.</em></p>"
    verdict = "PASS" if val.passed else "REJECT"
    color = "#2da44e" if val.passed else "#d1242f"
    rows = ""
    for c in val.checks:
        mark = "PASS" if c.passed else "FAIL"
        mc = "#2da44e" if c.passed else "#d1242f"
        crit = "critical" if c.critical else "info"
        rows += (f"<tr><td><b style='color:{mc}'>{mark}</b></td>"
                 f"<td>{_html.escape(c.name)}</td><td>{crit}</td>"
                 f"<td>{_html.escape(c.detail)}</td></tr>")
    return (f"<h2>Validation verdict: <span style='color:{color}'>{verdict}</span></h2>"
            f"<table><tr><th>Result</th><th>Check</th><th>Type</th><th>Detail</th></tr>"
            f"{rows}</table>")


def _journal_summary(journal: pd.DataFrame) -> str:
    if journal is None or journal.empty:
        return "<p><em>No trades.</em></p>"
    j = journal
    ret = j["return"].astype(float)
    summ = {
        "Round trips": len(j),
        "Win rate": f"{(ret > 0).mean():.1%}",
        "Avg return / trade": f"{ret.mean():.2%}",
        "Median return / trade": f"{ret.median():.2%}",
        "Best trade": f"{ret.max():.2%}",
        "Worst trade": f"{ret.min():.2%}",
        "Avg holding (days)": f"{j['holding_days'].dropna().astype(float).mean():.0f}",
    }
    top = j.reindex(ret.sort_values(ascending=False).index).head(5)
    rows = "".join(
        f"<tr><td>{_html.escape(str(row['symbol']))}</td>"
        f"<td>{pd.to_datetime(row['entry_date']).date()}</td>"
        f"<td>{pd.to_datetime(row['exit_date']).date()}</td>"
        f"<td>{float(row['return']):.2%}</td>"
        f"<td>{'' if pd.isna(row['holding_days']) else int(row['holding_days'])}</td></tr>"
        for _, row in top.iterrows()
    )
    return (_dict_table("Trade journal summary", summ) +
            "<h3>Top 5 trades</h3><table>"
            "<tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>Return</th><th>Days</th></tr>"
            f"{rows}</table>")


_CSS = """
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   max-width:960px;margin:24px auto;padding:0 16px;color:#1f2328;line-height:1.5}
 h1{margin-bottom:4px} h2{margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:4px}
 table{border-collapse:collapse;margin:8px 0 16px;width:100%}
 td,th{border:1px solid #d0d7de;padding:6px 10px;text-align:left;font-size:14px}
 th{background:#f6f8fa} img{max-width:100%;margin:8px 0;border:1px solid #eaeef2;border-radius:6px}
 .banner{padding:10px 14px;border-radius:8px;margin:12px 0;font-size:14px}
 .syn{background:#fff8c5;border:1px solid #d4a72c} .real{background:#dafbe1;border:1px solid #2da44e}
 .note{color:#57606a;font-size:13px;margin-top:32px}
</style>
"""


def generate_report(
    res,
    out_path: str,
    title: str = "Experiment",
    validation=None,
    bench_metrics: dict | None = None,
    data_kind: str = "synthetic",      # "synthetic" or "real"
    provenance: dict | None = None,
) -> str:
    """Write a self-contained HTML report. Returns the path."""
    eq = res.equity
    banner_cls = "real" if data_kind == "real" else "syn"
    banner_txt = ("REAL historical market data — metrics are research results "
                  "(still validate further before trading)." if data_kind == "real"
                  else "SYNTHETIC data — these numbers test the pipeline only and do "
                       "NOT imply a profitable strategy.")
    prov = provenance or (validation.provenance if validation else {})
    prov = prov or {"generated": dt.datetime.now().isoformat(timespec="seconds")}

    body = [
        _CSS,
        f"<h1>{_html.escape(title)}</h1>",
        f"<div class='banner {banner_cls}'><b>Scope:</b> {banner_txt}</div>",
        _dict_table("Provenance", prov),
        _validation_table(validation),
        "<h2>Performance &amp; trading metrics</h2>",
        _metrics_table(res.metrics),
    ]
    if bench_metrics:
        body.append("<h2>Benchmark-relative</h2>")
        body.append(_dict_table("Vs benchmark", bench_metrics))
    body += [
        "<h2>Charts</h2>",
        _equity_chart(eq),
        _drawdown_chart(eq),
        _attribution_chart(getattr(res, "importance", None)),
        "<h2>Trades</h2>",
        _journal_summary(getattr(res, "journal", None)),
        "<p class='note'>Generated by the Quant Engine research platform. "
        "Research scaffolding, not investment advice.</p>",
    ]
    html_doc = "<!doctype html><html><meta charset='utf-8'><body>" + "".join(body) + "</body></html>"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path
