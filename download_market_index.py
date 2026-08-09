"""Download the unadjusted CSI 300 daily close used by Stage-4 context analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import PARQUET_COMPRESSION, STAGE4_REPORT_DIR


def fetch_index(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one index through a dedicated Baostock login."""
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("baostock is required") from exc
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        result = bs.query_history_k_data_plus(code, "date,code,close",
            start_date=start, end_date=end, frequency="d", adjustflag="3")
        if result.error_code != "0":
            raise RuntimeError(result.error_msg)
        rows = []
        while result.next():
            rows.append(result.get_row_data())
    finally:
        bs.logout()
    frame = pd.DataFrame(rows, columns=["date", "code", "close"])
    frame["date"] = pd.to_datetime(frame["date"]); frame["close"] = pd.to_numeric(frame["close"])
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="sh.000300")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output", type=Path, default=STAGE4_REPORT_DIR / "sh.000300_daily.parquet")
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    fetch_index(args.code, args.start, args.end).to_parquet(
        args.output, index=False, compression=PARQUET_COMPRESSION)
    print(args.output)


if __name__ == "__main__":
    main()
