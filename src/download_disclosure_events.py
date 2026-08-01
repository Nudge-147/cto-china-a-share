"""Download all-market announcement dates needed for the CTO news-window test.

Uses the same Eastmoney endpoints wrapped by AkShare, but requests their raw
records so that the original announcement and ex-date fields are preserved.
The script is resumable at the (event type, financial report date) level.

Inputs: AkShare/Eastmoney announcement endpoints.
Outputs: checkpointed event-date partitions under ``data/cto_baostock/disclosures``.
Role: data stage for the Week-4 information-versus-attention label extension.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests


OUT = Path("data/cto_baostock/disclosures")
MANIFEST = OUT / "manifest.json"
PAGE_SIZE = 500
MAX_RETRIES = 3


SPECS = {
    "earnings_forecast": {
        "url": "https://datacenter.eastmoney.com/securities/api/data/v1/get",
        "report_name": "RPT_PUBLIC_OP_NEWPREDICT",
        "sort_columns": "NOTICE_DATE,SECURITY_CODE",
        "sort_types": "-1,-1",
        "report_field": "REPORT_DATE",
        "date_fields": {"announcement": "NOTICE_DATE"},
    },
    "periodic_report": {
        "url": "https://datacenter-web.eastmoney.com/api/data/v1/get",
        "report_name": "RPT_LICO_FN_CPD",
        "sort_columns": "UPDATE_DATE,SECURITY_CODE",
        "sort_types": "-1,-1",
        "report_field": "REPORTDATE",
        "date_fields": {"announcement": "NOTICE_DATE"},
    },
    "earnings_flash": {
        "url": "https://datacenter.eastmoney.com/securities/api/data/v1/get",
        "report_name": "RPT_FCI_PERFORMANCEE",
        "sort_columns": "UPDATE_DATE,SECURITY_CODE",
        "sort_types": "-1,-1",
        "report_field": "REPORT_DATE",
        "date_fields": {"announcement": "NOTICE_DATE"},
        "extra_filter": '(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")',
    },
    "dividend": {
        "url": "https://datacenter-web.eastmoney.com/api/data/v1/get",
        "report_name": "RPT_SHAREBONUS_DET",
        "sort_columns": "PLAN_NOTICE_DATE",
        "sort_types": "-1",
        "report_field": "REPORT_DATE",
        "date_fields": {
            "plan_announcement": "PLAN_NOTICE_DATE",
            "announcement": "NOTICE_DATE",
            "ex_date": "EX_DIVIDEND_DATE",
        },
        "extra_params": {"source": "WEB", "client": "WEB"},
    },
}


def report_dates(start_year: int, end_year: int) -> list[str]:
    end = date.today()
    dates: list[str] = []
    for year in range(start_year, end_year + 1):
        for month_day in ("0331", "0630", "0930", "1231"):
            value = f"{year}{month_day}"
            if pd.Timestamp(value) <= pd.Timestamp(end):
                dates.append(value)
    return dates


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"completed": {}, "failures": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def request_json(session: requests.Session, url: str, params: dict) -> dict:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            # The provider occasionally leaves a connection open indefinitely.
            # Fail this page quickly and let the resumable manifest move on.
            response = session.get(url, params=params, timeout=(10, 20))
            response.raise_for_status()
            payload = response.json()
            # Eastmoney uses code 9201 / result=None for a genuinely empty
            # report period (common for interim dividend tables).  This is an
            # observed zero, not a failed announcement download.
            if payload.get("success") is False and payload.get("code") == 9201:
                return {"result": {"pages": 0, "count": 0, "data": []}}
            if not payload.get("success", True) or not payload.get("result"):
                raise ValueError(f"unexpected API payload: {str(payload)[:300]}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(min(30, 1.5 * (2**attempt)) + random.uniform(0, 0.8))
    raise RuntimeError(f"request failed after {MAX_RETRIES} attempts: {last_error}")


def fetch_one(session: requests.Session, event_type: str, report_date: str) -> pd.DataFrame:
    spec = SPECS[event_type]
    report_iso = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}"
    base_filter = f"({spec['report_field']}='{report_iso}')"
    filter_ = f"{spec.get('extra_filter', '')}{base_filter}"
    params = {
        "sortColumns": spec["sort_columns"],
        "sortTypes": spec["sort_types"],
        "pageSize": str(PAGE_SIZE),
        "pageNumber": "1",
        "reportName": spec["report_name"],
        "columns": "ALL",
        "filter": filter_,
        **spec.get("extra_params", {}),
    }
    first = request_json(session, spec["url"], params)["result"]
    pages = int(first.get("pages") or 0)
    parts = [pd.DataFrame(first.get("data") or [])]
    for page in range(2, pages + 1):
        params["pageNumber"] = str(page)
        parts.append(pd.DataFrame(request_json(session, spec["url"], params)["result"].get("data") or []))
        time.sleep(random.uniform(0.08, 0.20))
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if raw.empty:
        return pd.DataFrame(columns=["stock_code", "event_type", "event_date", "event_role", "report_date", "source"])
    code = raw.get("SECURITY_CODE", pd.Series(index=raw.index, dtype="object")).astype(str).str.zfill(6)
    events = []
    for role, field in spec["date_fields"].items():
        if field not in raw:
            continue
        x = pd.DataFrame({
            "stock_code": code,
            "event_type": event_type,
            "event_date": pd.to_datetime(raw[field], errors="coerce").dt.normalize(),
            "event_role": role,
            "report_date": report_iso,
            "source": "AkShare/Eastmoney",
        })
        events.append(x)
    result = pd.concat(events, ignore_index=True)
    return result.dropna(subset=["event_date"]).drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--types", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--refresh", action="store_true", help="re-fetch completed partitions")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    dates = report_dates(args.start_year, args.end_year)
    total = len(dates) * len(args.types)
    done = 0
    for event_type in args.types:
        folder = OUT / event_type
        folder.mkdir(exist_ok=True)
        for report_date in dates:
            key = f"{event_type}/{report_date}"
            target = folder / f"{report_date}.csv.gz"
            if target.exists() and key in manifest["completed"] and not args.refresh:
                done += 1
                continue
            try:
                print(f"[{done + 1}/{total}] START {key}", flush=True)
                events = fetch_one(session, event_type, report_date)
                events.to_csv(target, index=False, compression="gzip")
                manifest["completed"][key] = {"rows": int(len(events)), "downloaded_at": pd.Timestamp.now().isoformat()}
                manifest["failures"].pop(key, None)
                save_manifest(manifest)
                done += 1
                print(f"[{done}/{total}] {key}: {len(events)} event dates", flush=True)
            except Exception as exc:
                manifest["failures"][key] = str(exc)
                save_manifest(manifest)
                print(f"[{done}/{total}] {key}: FAILED {exc}", flush=True)
            time.sleep(random.uniform(0.25, 0.60))


if __name__ == "__main__":
    main()
