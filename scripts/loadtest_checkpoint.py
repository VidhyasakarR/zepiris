#!/usr/bin/env python
"""Concurrency ladder against /v1/checkpoint/verify with real go-online selfies.

loadtest_ramp.py measures facematch only. The go-online check runs three
models per request (face + liveness, then the dress-code classifier and the
logo reader), so its capacity has to be measured on that endpoint, with real
captures rather than a synthesized probe: the detector and the dress-code crop
behave differently on a phone selfie than on a gallery portrait.

Each level sends ``--requests-per-level`` requests with exactly ``level`` in
flight. The ladder stops early once a level's error rate passes
``--stop-error-rate``, so it never keeps hammering a box that is already
shedding.

Usage:
    python scripts/loadtest_checkpoint.py --base-url https://host \
        --images-dir path/to/selfies --reference path/to/profile.jpg \
        --levels 1,2,4,8,16 --requests-per-level 50 --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered)) - 1))
    return ordered[index]


async def run_level(client, url, bodies, level, total):
    """Send ``total`` requests keeping ``level`` in flight; return per-request results."""
    results = []
    next_index = 0

    async def worker():
        nonlocal next_index
        while next_index < total:
            body = bodies[next_index % len(bodies)]
            next_index += 1
            start = time.perf_counter()
            try:
                response = await client.post(url, content=body,
                                             headers={"Content-Type": "application/json"})
                status = response.status_code
            except httpx.HTTPError as exc:
                status = type(exc).__name__
            results.append({"ms": (time.perf_counter() - start) * 1000, "status": status})

    wall_start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(level)))
    return results, time.perf_counter() - wall_start


def summarise(level, results, wall_s):
    ok = [r["ms"] for r in results if r["status"] == 200]
    errors: dict[str, int] = {}
    for r in results:
        if r["status"] != 200:
            errors[str(r["status"])] = errors.get(str(r["status"]), 0) + 1
    return {
        "concurrency": level,
        "requests": len(results),
        "ok": len(ok),
        "error_rate": round(1 - len(ok) / len(results), 4) if results else 0,
        "errors": errors,
        "wall_s": round(wall_s, 2),
        "throughput_rps": round(len(ok) / wall_s, 2) if wall_s else 0,
        "images_per_min": round(len(ok) / wall_s * 60, 1) if wall_s else 0,
        "min_ms": round(min(ok), 1) if ok else None,
        "avg_ms": round(statistics.mean(ok), 1) if ok else None,
        "p50_ms": round(percentile(ok, 50), 1),
        "p90_ms": round(percentile(ok, 90), 1),
        "p95_ms": round(percentile(ok, 95), 1),
        "p99_ms": round(percentile(ok, 99), 1),
        "max_ms": round(max(ok), 1) if ok else None,
    }


async def main(args):
    reference = base64.b64encode(Path(args.reference).read_bytes()).decode()
    selfies = sorted(p for p in Path(args.images_dir).glob("*.jpg") if p.name != Path(args.reference).name)
    # Pre-encode every body once, so the client's own JSON/base64 work is not
    # part of the measured latency.
    bodies = [json.dumps({"face_check_b64": base64.b64encode(p.read_bytes()).decode(),
                          "source_selfie_b64": reference}).encode() for p in selfies]
    url = args.base_url.rstrip("/") + "/v1/checkpoint/verify"
    levels = [int(x) for x in args.levels.split(",")]
    summaries = []
    limits = httpx.Limits(max_connections=max(levels) + 4, max_keepalive_connections=max(levels) + 4)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        await run_level(client, url, bodies, 1, 3)  # warm-up, not reported
        print(f"{len(bodies)} distinct selfies -> {url}")
        print(f"{'conc':>4} {'ok':>4} {'err%':>5} {'img/s':>6} {'img/min':>7} {'min':>6} {'avg':>6} "
              f"{'p50':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6}")
        for level in levels:
            results, wall_s = await run_level(client, url, bodies, level, args.requests_per_level)
            s = summarise(level, results, wall_s)
            summaries.append(s)
            print(f"{s['concurrency']:>4} {s['ok']:>4} {s['error_rate']*100:>5.1f} {s['throughput_rps']:>6} "
                  f"{s['images_per_min']:>7} {s['min_ms']!s:>6} {s['avg_ms']!s:>6} {s['p50_ms']:>6} "
                  f"{s['p90_ms']:>6} {s['p95_ms']:>6} {s['p99_ms']:>6} {s['max_ms']!s:>6} {s['errors'] or ''}",
                  flush=True)
            if s["error_rate"] > args.stop_error_rate:
                print(f"stopping: error rate {s['error_rate']:.0%} above {args.stop_error_rate:.0%}")
                break
    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--requests-per-level", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--stop-error-rate", type=float, default=0.2)
    ap.add_argument("--json")
    asyncio.run(main(ap.parse_args()))
