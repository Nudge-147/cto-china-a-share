"""Checkpointed Baostock 5-minute and unadjusted-daily downloader."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import baostock as bs
import pandas as pd

from config import (
    BACKOFF_BASE_SECONDS, COVERAGE_GAP_YEAR, COVERAGE_PROBE_STOCKS,
    DAILY_REFERENCE_DIR, DAILY_REFERENCE_FIELDS, DEFAULT_WORKERS,
    DOWNLOAD_LOG_PATH, DOWNLOAD_START_DATE, FAILED_LOG_PATH, MINUTE_FIELDS,
    MINUTE_FREQUENCY, PARQUET_COMPRESSION, PROGRESS_PATH, PROGRESS_VERSION,
    RAW_5MIN_DIR, REQUEST_SLEEP_SECONDS, RETRY_ATTEMPTS, SEGMENT_5MIN_DIR,
    STOCK_LIST_PATH, UNADJUSTED_FLAG,
)


LOGGER = logging.getLogger("download_5min")
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def setup_logging() -> None:
    """Log to both console and a persistent file."""
    DOWNLOAD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers = [logging.StreamHandler(), logging.FileHandler(DOWNLOAD_LOG_PATH, encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def load_stock_codes(path: Path, limit: int | None) -> list[str]:
    """Read and validate the one-column stock universe."""
    stocks = pd.read_csv(path, dtype=str)
    if list(stocks.columns) != ["code"]:
        raise ValueError(f"{path} must contain exactly one code column")
    codes = stocks["code"].dropna().str.strip()
    if not codes.str.fullmatch(r"(?:sh|sz)\.\d{6}").all():
        raise ValueError("stock_list.csv contains invalid Baostock codes")
    result = codes.drop_duplicates().tolist()
    return result[:limit] if limit else result


def load_progress() -> dict[str, Any]:
    """Load progress or create a new state document."""
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {"version": PROGRESS_VERSION, "stocks": {}, "coverage": {}}


def save_progress(progress: dict[str, Any]) -> None:
    """Atomically persist parent-owned progress state."""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now().astimezone().isoformat()
    temporary = PROGRESS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, PROGRESS_PATH)


def append_failure(item: dict[str, Any]) -> None:
    """Append one machine-readable failure without stopping the batch."""
    FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged_at": datetime.now().astimezone().isoformat(), **item}
    with FAILED_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def login_or_raise() -> None:
    """Open a Baostock session in the current worker process."""
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"login failed: {result.error_code} {result.error_msg}")


def initialize_worker() -> None:
    """Give each process its own independent Baostock login."""
    login_or_raise()


def reconnect_worker() -> None:
    """Reconnect a worker after a retryable API/session failure."""
    try:
        bs.logout()
    except Exception:
        pass
    login_or_raise()


def collect_response(response: Any) -> pd.DataFrame:
    """Consume one Baostock result cursor."""
    if response.error_code != "0":
        raise RuntimeError(f"{response.error_code}: {response.error_msg}")
    rows: list[list[str]] = []
    while response.next():
        rows.append(response.get_row_data())
    return pd.DataFrame(rows, columns=response.fields)


def with_retry(request: Callable[[], Any]) -> pd.DataFrame:
    """Run one request with exponential backoff and reconnection."""
    error: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            frame = collect_response(request())
            time.sleep(REQUEST_SLEEP_SECONDS)
            return frame
        except Exception as exc:
            error = exc
            if attempt == RETRY_ATTEMPTS - 1:
                break
            time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
            reconnect_worker()
    raise RuntimeError(f"request failed after {RETRY_ATTEMPTS} attempts: {error}")


def parse_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw strings into the canonical parquet schema."""
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["time"] = pd.to_datetime(
        frame["time"].astype(str).str[:14], format="%Y%m%d%H%M%S", errors="coerce"
    )
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["date", "time"]].isna().any().any():
        raise ValueError("unparseable date/time returned by Baostock")
    return frame[["date", "time", "code", *NUMERIC_COLUMNS]]


def parse_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert direct unadjusted daily rows and validate the adjustment flag."""
    frame = raw.copy()
    if not frame.empty and not frame["adjustflag"].eq(UNADJUSTED_FLAG).all():
        raise ValueError("daily reference contains non-raw adjustment flags")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["tradestatus"] = pd.to_numeric(frame["tradestatus"], errors="coerce").astype("Int8")
    return frame[["date", "code", *NUMERIC_COLUMNS, "tradestatus"]]


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write snappy parquet and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}.parquet")
    frame.to_parquet(temporary, index=False, compression=PARQUET_COMPRESSION)
    os.replace(temporary, path)


def query_minute_segment(code: str, year: int, end_date: str) -> pd.DataFrame:
    """Request one natural-year minute segment."""
    start = max(f"{year}-01-01", DOWNLOAD_START_DATE)
    end = min(f"{year}-12-31", end_date)
    request = lambda: bs.query_history_k_data_plus(
        code, MINUTE_FIELDS, start_date=start, end_date=end,
        frequency=MINUTE_FREQUENCY, adjustflag=UNADJUSTED_FLAG,
    )
    return parse_minutes(with_retry(request))


def download_segment_task(code: str, year: int, end_date: str) -> dict[str, Any]:
    """Worker task: download and save one stock-year segment."""
    started = time.perf_counter()
    try:
        frame = query_minute_segment(code, year, end_date)
        atomic_parquet(frame, SEGMENT_5MIN_DIR / f"{code}_{year}.parquet")
        return {"status": "empty" if frame.empty else "ok", "code": code, "year": year,
                "rows": len(frame), "first_date": date_value(frame, "min"),
                "last_date": date_value(frame, "max"), "seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"status": "failed", "code": code, "year": year,
                "error": repr(exc), "seconds": time.perf_counter() - started}


def date_value(frame: pd.DataFrame, operation: str) -> str | None:
    """Return a JSON-safe boundary date."""
    if frame.empty:
        return None
    value = frame["date"].min() if operation == "min" else frame["date"].max()
    return value.date().isoformat()


def download_daily_task(code: str, end_date: str) -> dict[str, Any]:
    """Worker task: download the standard unadjusted daily companion table."""
    started = time.perf_counter()
    try:
        request = lambda: bs.query_history_k_data_plus(
            code, DAILY_REFERENCE_FIELDS, start_date=DOWNLOAD_START_DATE,
            end_date=end_date, frequency="d", adjustflag=UNADJUSTED_FLAG,
        )
        frame = parse_daily(with_retry(request))
        atomic_parquet(frame, DAILY_REFERENCE_DIR / f"{code}.parquet")
        return {"status": "ok", "code": code, "rows": len(frame),
                "first_date": date_value(frame, "min"), "last_date": date_value(frame, "max"),
                "seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"status": "failed", "code": code, "error": repr(exc),
                "seconds": time.perf_counter() - started}


def run_tasks(executor: ProcessPoolExecutor, function: Callable[..., dict[str, Any]], tasks: list[tuple[Any, ...]], on_result: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Consume and checkpoint each task immediately on completion."""
    futures = {executor.submit(function, *task): task for task in tasks}
    results: list[dict[str, Any]] = []
    for future in as_completed(futures):
        try:
            item = future.result()
        except Exception as exc:
            task = futures[future]
            item = {"status": "failed", "code": str(task[0]), "task": list(task), "error": repr(exc)}
            if len(task) > 1 and isinstance(task[1], int):
                item["year"] = task[1]
        results.append(item)
        if on_result:
            on_result(item)
    return results


def record_daily_results(progress: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Record daily-reference outcomes and log row counts/timing."""
    for item in results:
        code = item.get("code", "worker")
        progress["stocks"].setdefault(code, {})["daily_reference"] = item
        if item["status"] == "failed":
            append_failure({"kind": "daily_reference", **item})
            LOGGER.error("daily %s failed: %s", code, item.get("error"))
        else:
            LOGGER.info("daily %s rows=%s seconds=%.1f", code, item["rows"], item["seconds"])
        save_progress(progress)


def record_segment_results(progress: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Record completed year segments in the parent-owned checkpoint."""
    for item in results:
        code, year = item.get("code", "worker"), str(item.get("year", "unknown"))
        stock = progress["stocks"].setdefault(code, {})
        stock.setdefault("years", {})[year] = item
        if item["status"] == "failed":
            append_failure({"kind": "minute_segment", **item})
            LOGGER.error("minute %s %s failed: %s", code, year, item.get("error"))
        else:
            LOGGER.info("minute %s %s rows=%s seconds=%.1f", code, year, item["rows"], item["seconds"])
        save_progress(progress)


def segment_is_complete(progress: dict[str, Any], code: str, year: int, current_year: int) -> bool:
    """Skip completed historical segments, but always refresh the current year."""
    if year == current_year:
        return False
    item = progress.get("stocks", {}).get(code, {}).get("years", {}).get(str(year), {})
    path = SEGMENT_5MIN_DIR / f"{code}_{year}.parquet"
    if item.get("status") == "coverage_skipped":
        return True
    return item.get("status") in {"ok", "empty"} and path.exists()


def year_tasks(codes: list[str], years: list[int], end_date: str, progress: dict[str, Any]) -> list[tuple[str, int, str]]:
    """Build pending stock-year tasks from the checkpoint."""
    current_year = pd.Timestamp(end_date).year
    return [(code, year, end_date) for code in codes for year in years
            if not segment_is_complete(progress, code, year, current_year)]


def decide_coverage_skip(codes: list[str], results: list[dict[str, Any]], progress: dict[str, Any]) -> bool:
    """Skip later 2019 requests only after 20 successful empty probes."""
    if len(codes) < COVERAGE_PROBE_STOCKS:
        return False
    by_code = {item.get("code"): item for item in results}
    tested = [by_code.get(code) or progress["stocks"].get(code, {}).get("years", {}).get(str(COVERAGE_GAP_YEAR), {})
              for code in codes[:COVERAGE_PROBE_STOCKS]]
    return all(item.get("status") == "empty" for item in tested)


def mark_coverage_skips(progress: dict[str, Any], codes: list[str]) -> None:
    """Checkpoint automatically skipped 2019 segments."""
    for code in codes:
        item = {"status": "coverage_skipped", "code": code, "year": COVERAGE_GAP_YEAR, "rows": 0}
        progress["stocks"].setdefault(code, {}).setdefault("years", {})[str(COVERAGE_GAP_YEAR)] = item
    save_progress(progress)


def merge_stock(code: str, years: list[int], progress: dict[str, Any]) -> None:
    """Merge all available yearly files once into the final per-stock parquet."""
    paths = [SEGMENT_5MIN_DIR / f"{code}_{year}.parquet" for year in years]
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        LOGGER.error("merge %s skipped: no segment files", code)
        return
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(["code", "date", "time"]).sort_values("time")
    atomic_parquet(merged, RAW_5MIN_DIR / f"{code}.parquet")
    failed = [str(year) for year in years if progress["stocks"].get(code, {}).get("years", {}).get(str(year), {}).get("status") == "failed"]
    status = "partial" if failed else "ok"
    progress["stocks"].setdefault(code, {})["merged"] = {
        "status": status, "rows": len(merged), "failed_years": failed,
        "first_date": date_value(merged, "min"), "last_date": date_value(merged, "max"),
    }
    LOGGER.info("merged %s rows=%s status=%s", code, len(merged), status)
    save_progress(progress)


def run_download(codes: list[str], workers: int, end_date: str) -> None:
    """Orchestrate daily references, coverage probes, year downloads, and merging."""
    progress = load_progress()
    final_year = pd.Timestamp(end_date).year
    years = list(range(pd.Timestamp(DOWNLOAD_START_DATE).year, final_year + 1))
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as executor:
        run_tasks(executor, download_daily_task, [(code, end_date) for code in codes],
                  lambda item: record_daily_results(progress, [item]))
        probes = codes[:min(COVERAGE_PROBE_STOCKS, len(codes))]
        results = run_tasks(executor, download_segment_task,
                            year_tasks(probes, [COVERAGE_GAP_YEAR], end_date, progress),
                            lambda item: record_segment_results(progress, [item]))
        skip_coverage = decide_coverage_skip(probes, results, progress)
        progress["coverage"][str(COVERAGE_GAP_YEAR)] = {"probed": probes, "all_empty": skip_coverage}
        remaining = codes[len(probes):]
        if skip_coverage:
            mark_coverage_skips(progress, remaining)
        else:
            run_tasks(executor, download_segment_task,
                      year_tasks(remaining, [COVERAGE_GAP_YEAR], end_date, progress),
                      lambda item: record_segment_results(progress, [item]))
        later_years = [year for year in years if year != COVERAGE_GAP_YEAR]
        run_tasks(executor, download_segment_task, year_tasks(codes, later_years, end_date, progress),
                  lambda item: record_segment_results(progress, [item]))
    for code in codes:
        merge_stock(code, years, progress)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--stock-list", type=Path, default=STOCK_LIST_PATH)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    setup_logging()
    codes = load_stock_codes(args.stock_list, args.limit)
    if args.merge_only:
        progress = load_progress()
        years = list(range(pd.Timestamp(DOWNLOAD_START_DATE).year,
                           pd.Timestamp(args.end_date).year + 1))
        for code in codes:
            merge_stock(code, years, progress)
        return
    LOGGER.info("starting stocks=%s workers=%s end=%s", len(codes), args.workers, args.end_date)
    run_download(codes, args.workers, args.end_date)


if __name__ == "__main__":
    main()
