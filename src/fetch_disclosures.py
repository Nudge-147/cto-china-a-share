"""Probe AkShare disclosure endpoints before the production event downloader.

Inputs: a small hard-coded endpoint/date smoke-test set.
Outputs: local schema samples under ``data/``.
Role: exploratory API validation only; use ``download_disclosure_events.py`` for research data.
"""

from pathlib import Path
import json
import inspect
import signal
import akshare as ak
import pandas as pd

OUT = Path("data")
OUT.mkdir(exist_ok=True)

def attempt(name, fn, **kwargs):
    result = {"function": name, "kwargs": kwargs, "status": "failed"}
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("request timed out after 20s")))
        signal.alarm(20)
        df = fn(**kwargs)
        signal.alarm(0)
        result.update({"status": "ok", "rows": int(len(df)), "columns": list(df.columns)})
        if len(df):
            result["sample"] = df.head(3).astype(str).to_dict(orient="records")
            df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        signal.alarm(0)
        result["error"] = repr(exc)
    return result

def main():
    results = []
    # One current and one older period: this tests both reach and schema.
    for date in ["20250331", "20190331"]:
        results.append(attempt("stock_yjyg_em_" + date, ak.stock_yjyg_em, date=date))
    results.append(attempt("stock_zh_a_disclosure_report_cninfo", ak.stock_zh_a_disclosure_report_cninfo,
                          symbol="000001", market="沪深京", category="业绩预告",
                          start_date="20190101", end_date="20260722"))
    results.append(attempt("stock_individual_notice_report", ak.stock_individual_notice_report,
                          security="000001", symbol="全部", begin_date="20190101", end_date="20260722"))
    results.append(attempt("stock_notice_report", ak.stock_notice_report,
                          symbol="全部", date="20250722"))
    Path(OUT / "disclosure_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
