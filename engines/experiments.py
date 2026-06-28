"""
Experiment Manager
==================
A lightweight ledger so every run is comparable instead of an isolated HTML file.
Deliberately NOT a heavy database — two flat files:

  reports/experiments.csv    one row per run (the comparison table)
  reports/experiments.jsonl  one JSON line per run (full record incl. config hash)

Plus an auto-generated reports/index.html that lists every experiment with its
metrics, validation verdict, and a link to its report.

The goal is enabling comparison across hundreds of experiments — "which config
beat which, and did it pass?" — not persistence for its own sake.

Research scaffolding, not investment advice.

Dependencies:
    pip install pandas
"""
from __future__ import annotations

import json
import datetime as dt
import html as _html
from pathlib import Path

import pandas as pd

import data_engine as de

REPORTS_DIR = de.ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = REPORTS_DIR / "experiments.csv"
JSONL_PATH = REPORTS_DIR / "experiments.jsonl"
INDEX_PATH = REPORTS_DIR / "index.html"

# Columns shown in the comparison table (CSV)
FIELDS = [
    "run_id", "timestamp", "strategy", "universe", "n_symbols", "period",
    "data_kind", "sharpe", "cagr", "max_drawdown", "turnover",
    "validation", "config_hash", "report",
]


def _next_run_id() -> str:
    if CSV_PATH.exists():
        try:
            n = len(pd.read_csv(CSV_PATH))
        except Exception:  # noqa: BLE001
            n = 0
    else:
        n = 0
    return f"{n + 1:03d}"


def register(
    *,
    strategy: str,
    metrics: dict,
    report_path: str,
    universe: str = "",
    n_symbols: int = 0,
    period: str = "",
    data_kind: str = "synthetic",
    validation: str | bool | None = None,
    config_hash: str = "",
    full_record: dict | None = None,
) -> dict:
    """Append one run to the ledger and rebuild index.html. Returns the row."""
    perf = metrics.get("Performance", {}) if metrics else {}
    trad = metrics.get("Trading", {}) if metrics else {}
    verdict = ("PASS" if validation is True else
               "FAIL" if validation is False else
               (validation if isinstance(validation, str) else "—"))
    row = {
        "run_id": _next_run_id(),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "universe": universe,
        "n_symbols": n_symbols,
        "period": period,
        "data_kind": data_kind,
        "sharpe": perf.get("Sharpe"),
        "cagr": perf.get("CAGR"),
        "max_drawdown": perf.get("Max Drawdown"),
        "turnover": trad.get("Turnover (ann)"),
        "validation": verdict,
        "config_hash": config_hash,
        "report": Path(report_path).name,
    }

    # append CSV
    df_row = pd.DataFrame([{k: row.get(k) for k in FIELDS}])
    header = not CSV_PATH.exists()
    df_row.to_csv(CSV_PATH, mode="a", header=header, index=False)

    # append full JSONL
    rec = dict(row)
    if full_record:
        rec["full"] = full_record
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")

    build_index()
    return row


def load_index() -> pd.DataFrame:
    """The comparison table across all experiments."""
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(CSV_PATH)


def compare(sort_by: str = "sharpe", ascending: bool = False) -> pd.DataFrame:
    df = load_index()
    if sort_by in df.columns and not df.empty:
        df = df.sort_values(sort_by, ascending=ascending)
    return df


def build_index() -> str:
    """Render reports/index.html: a sortable-looking table linking each report."""
    df = load_index()
    rows = ""
    for _, r in df.iterrows():
        verdict = str(r.get("validation", "—"))
        vc = "#2da44e" if verdict == "PASS" else "#d1242f" if verdict == "FAIL" else "#57606a"
        dk = str(r.get("data_kind", ""))
        dkc = "#2da44e" if dk == "real" else "#9a6700"
        report = _html.escape(str(r.get("report", "")))
        link = f"<a href='{report}'>{report}</a>" if report and report != "nan" else "—"
        rows += (
            "<tr>"
            f"<td>{_html.escape(str(r.get('run_id','')))}</td>"
            f"<td>{_html.escape(str(r.get('strategy','')))}</td>"
            f"<td>{_html.escape(str(r.get('universe','')))}</td>"
            f"<td style='color:{dkc}'>{_html.escape(dk)}</td>"
            f"<td>{_html.escape(str(r.get('sharpe','')))}</td>"
            f"<td>{_html.escape(str(r.get('cagr','')))}</td>"
            f"<td>{_html.escape(str(r.get('max_drawdown','')))}</td>"
            f"<td>{_html.escape(str(r.get('turnover','')))}</td>"
            f"<td style='color:{vc};font-weight:600'>{_html.escape(verdict)}</td>"
            f"<td>{_html.escape(str(r.get('timestamp','')))}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    css = (
        "<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
        "max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328}"
        "table{border-collapse:collapse;width:100%;font-size:14px}"
        "td,th{border:1px solid #d0d7de;padding:6px 10px;text-align:left}"
        "th{background:#f6f8fa}h1{margin-bottom:2px}.note{color:#57606a;font-size:13px}</style>"
    )
    head = ("<tr><th>Run</th><th>Strategy</th><th>Universe</th><th>Data</th>"
            "<th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Turnover</th>"
            "<th>Validation</th><th>Timestamp</th><th>Report</th></tr>")
    doc = (
        "<!doctype html><html><meta charset='utf-8'><body>" + css +
        "<h1>Experiment Index</h1>"
        f"<p class='note'>{len(df)} experiments. "
        "Synthetic rows test the pipeline; only <b>real</b> rows are research results.</p>"
        f"<table>{head}{rows}</table>"
        "<p class='note'>Quant Engine research platform — research scaffolding, not investment advice.</p>"
        "</body></html>"
    )
    INDEX_PATH.write_text(doc, encoding="utf-8")
    return str(INDEX_PATH)


if __name__ == "__main__":
    print(load_index().to_string(index=False) if not load_index().empty
          else "No experiments yet. Run runner.py to create some.")
