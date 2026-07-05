"""Live training dashboard server.

    uv run python train/dashboard.py --run runs/night-3b [--port 8787]

Serves dashboard.html plus:
  /data          JSON: trainer metrics, per-call rollout summaries, error mix,
                 sample sheet names, baseline (if present), run config
  /samples/<f>   contact-sheet PNGs

Reads are incremental-friendly (files are append-only JSONL); the page polls
every 5s.
"""

from __future__ import annotations

import argparse
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows[-limit:] if limit else rows


def collect(run_dir: Path) -> dict:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    rollouts = read_jsonl(run_dir / "rollouts.jsonl")

    summaries = [r for r in rollouts if r.get("kind") == "group_summary"]
    per_sample = [r for r in rollouts if r.get("kind") != "group_summary"]

    # similarity percentiles + ink per call (from per-sample rows)
    by_call: dict[int, list[dict]] = {}
    for r in per_sample:
        by_call.setdefault(r["call"], []).append(r)
    sim_series = []
    for call in sorted(by_call):
        rows = by_call[call]
        sims = sorted(r["sim"] for r in rows if r.get("sim") is not None)
        inks = [r["ink_frac"] for r in rows if r.get("ink_frac") is not None]
        lens = [r["completion_chars"] for r in rows]
        if sims:
            q = lambda p: sims[min(len(sims) - 1, int(p * len(sims)))]
            sim_series.append({
                "call": call,
                "p25": q(0.25), "p50": q(0.50), "p75": q(0.75),
                "ink": sum(inks) / len(inks) if inks else None,
                "mean_completion_chars": sum(lens) / len(lens),
            })

    errors: dict[str, int] = {}
    for r in per_sample[-400:]:
        if r.get("error"):
            errors[r["error"]] = errors.get(r["error"], 0) + 1

    samples = sorted(
        (p.name for p in (run_dir / "samples").glob("*.png")), reverse=True
    )[:12]

    baseline = {}
    bl = run_dir.parent / "baseline.json"
    if bl.exists():
        baseline = json.loads(bl.read_text())

    config = {}
    cfg = run_dir / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        config = {k: c.get(k) for k in
                  ("model", "iters", "group_size", "batch_size", "temperature",
                   "learning_rate", "max_completion_length", "beta")}

    return {
        "run": run_dir.name,
        "config": config,
        "metrics": metrics,
        "summaries": summaries,
        "sim_series": sim_series,
        "errors": errors,
        "samples": samples,
        "baseline": baseline,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    run_dir = Path(args.run).resolve()

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send((HERE / "dashboard.html").read_bytes(),
                           "text/html; charset=utf-8")
            elif self.path == "/data":
                self._send(json.dumps(collect(run_dir)).encode(),
                           "application/json")
            elif self.path.startswith("/samples/"):
                p = run_dir / "samples" / Path(self.path).name
                if p.exists():
                    self._send(p.read_bytes(), "image/png")
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

    print(f"dashboard for {run_dir.name} -> http://localhost:{args.port}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
