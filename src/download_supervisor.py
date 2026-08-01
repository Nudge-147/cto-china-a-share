"""Keep Baostock shards alive for unattended overnight downloading.

The supervisor owns two worker processes.  It records a heartbeat every minute
and restarts both workers if either exits or no completed price pair is written
for STALL_SECONDS.  Workers are safe to restart because their downloader skips
already completed HFQ/raw file pairs.

Inputs: the Baostock downloader and its checkpointed daily-price directories.
Outputs: worker logs and a download heartbeat JSON under ``data/``.
Role: optional operational helper; not required for analytical replication.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "cto_baostock"
OUT = BASE / "outputs"
STATUS = OUT / "download_supervisor_status.json"


def completed_pairs() -> tuple[int, float]:
    hfq = {p.stem: p.stat().st_mtime for p in (BASE / "daily_hfq").glob("*.csv")}
    raw = {p.stem: p.stat().st_mtime for p in (BASE / "daily_raw").glob("*.csv")}
    common = set(hfq) & set(raw)
    latest = max((max(hfq[c], raw[c]) for c in common), default=0.0)
    return len(common), latest


def start_worker(shard: int, shards: int) -> subprocess.Popen:
    log = (OUT / f"worker_shard{shard:02d}.log").open("a", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-u", "baostock_pipeline.py", "--shard-index", str(shard), "--shard-count", str(shards)],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_workers(workers: list[subprocess.Popen]) -> None:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    deadline = time.monotonic() + 20
    for worker in workers:
        if worker.poll() is None:
            try:
                worker.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                worker.kill()


def write_status(workers: list[subprocess.Popen], pairs: int, latest: float, restarts: int, reason: str) -> None:
    STATUS.write_text(json.dumps({
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_pairs": pairs,
        "latest_pair_written_at_utc": datetime.fromtimestamp(latest, timezone.utc).isoformat() if latest else None,
        "worker_pids": [w.pid for w in workers],
        "worker_returncodes": [w.poll() for w in workers],
        "restart_count": restarts,
        "last_action": reason,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def supervise(shards: int, poll_seconds: int, stall_seconds: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workers = [start_worker(i, shards) for i in range(shards)]
    restarts = 0
    try:
        while True:
            pairs, latest = completed_pairs()
            exited = any(w.poll() is not None for w in workers)
            stalled = bool(latest and time.time() - latest > stall_seconds)
            reason = "healthy"
            if exited or stalled:
                reason = "worker_exited" if exited else "no_new_pair_within_stall_window"
                stop_workers(workers)
                workers = [start_worker(i, shards) for i in range(shards)]
                restarts += 1
            write_status(workers, pairs, latest, restarts, reason)
            time.sleep(poll_seconds)
    finally:
        stop_workers(workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stall-seconds", type=int, default=900)
    args = parser.parse_args()
    supervise(args.shards, args.poll_seconds, args.stall_seconds)
