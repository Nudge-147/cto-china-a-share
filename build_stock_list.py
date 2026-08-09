"""Build the 5-minute pilot universe from CTO raw daily prices."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from config import (
    CTO_DAILY_RAW_DIR,
    LIQUIDITY_SCAN_WORKERS,
    LIQUIDITY_START_DATE,
    LIQUIDITY_STOCK_COUNT,
    STOCK_LIST_PATH,
)


USE_COLUMNS = ["日期", "代码", "成交额", "交易状态"]


def summarize_stock(path: Path) -> dict[str, object] | None:
    """Return post-cutoff liquidity statistics for one stock file."""
    daily = pd.read_csv(path, usecols=USE_COLUMNS, encoding="utf-8-sig")
    if daily.empty:
        return None
    daily["日期"] = pd.to_datetime(daily["日期"], errors="coerce")
    daily["成交额"] = pd.to_numeric(daily["成交额"], errors="coerce")
    daily["交易状态"] = pd.to_numeric(daily["交易状态"], errors="coerce")
    cutoff = pd.Timestamp(LIQUIDITY_START_DATE)
    valid = daily[(daily["日期"] >= cutoff) & daily["交易状态"].eq(1)].dropna(subset=["成交额"])
    if valid.empty:
        return None
    return {
        "code": str(valid["代码"].iloc[-1]),
        "average_amount": float(valid["成交额"].mean()),
        "trading_days": int(len(valid)),
        "first_date": valid["日期"].min(),
        "last_date": valid["日期"].max(),
    }


def rank_stocks(paths: list[Path]) -> pd.DataFrame:
    """Scan daily files and rank stocks by mean daily traded amount."""
    with ThreadPoolExecutor(max_workers=LIQUIDITY_SCAN_WORKERS) as executor:
        summaries = list(executor.map(summarize_stock, paths))
    ranking = pd.DataFrame(item for item in summaries if item is not None)
    return ranking.sort_values(["average_amount", "code"], ascending=[False, True])


def write_stock_list(ranking: pd.DataFrame, output_path: Path) -> None:
    """Write exactly one code column for the top-N pilot universe."""
    selected = ranking.head(LIQUIDITY_STOCK_COUNT)[["code"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False, encoding="utf-8")


def main() -> None:
    paths = sorted(CTO_DAILY_RAW_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CTO daily files found in {CTO_DAILY_RAW_DIR}")
    ranking = rank_stocks(paths)
    if len(ranking) < LIQUIDITY_STOCK_COUNT:
        raise RuntimeError(f"Only {len(ranking)} stocks have usable liquidity data")
    write_stock_list(ranking, STOCK_LIST_PATH)
    top = ranking.head(LIQUIDITY_STOCK_COUNT)
    print(f"wrote {len(top)} stocks to {STOCK_LIST_PATH}")
    print(f"daily coverage: {top['first_date'].min().date()} to {top['last_date'].max().date()}")
    print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
