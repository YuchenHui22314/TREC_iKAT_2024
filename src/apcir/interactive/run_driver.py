"""CLI for the Sim.API driver.

Phase-0 use (read-only, NO conversation budget):
    python -m apcir.interactive.run_driver --preflight
        -> GET /auth/verify + GET /budget/check, prints team id + remaining budget.

Full driver (Phase 3+) drives the start/continue loop against our search server;
see `driver.py`. The token is read ONLY from env IKAT_SIM_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import sys

from .sim_client import SimClient, SimAPIError, DEFAULT_BASE_URL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="iKAT'26 Sim.API driver")
    p.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--mode", choices=["debug", "run"], default="debug")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="read-only: verify token + check budget, then exit (no conversation cost)",
    )
    # --- driver loop args (used in Phase 3+) ---
    p.add_argument("--server_url", default="http://127.0.0.1:8000",
                   help="URL of OUR search server")
    p.add_argument("--run_id", default=None)
    p.add_argument("--description", default="")
    # NOTE: the real /start schema has no track_persona field; if set, it is passed
    # through the free-form `extra` object (the API may ignore it). Verify in debug.
    p.add_argument("--track_persona", action="store_true", default=True)
    p.add_argument("--max_conversations", type=int, default=1)
    # safety: cap turns/session so a degenerate user-simulator loop can't run forever
    p.add_argument("--max_turns_per_session", type=int, default=30)
    p.add_argument("--results_dir",
                   default="/data/rech/huiyuche/TREC_iKAT_2024/results")
    # hard guard: official scored runs must be explicitly acknowledged
    p.add_argument("--i_understand_run_is_scored", action="store_true",
                   help="REQUIRED to use --mode run (protects the 6 official runs)")
    return p


def preflight(args) -> int:
    client = SimClient(base_url=args.base_url, mode=args.mode)
    try:
        team = client.verify()
        budget = client.budget_check()
    except SimAPIError as e:
        print(f"[preflight] FAILED: {e}", file=sys.stderr)
        return 1
    print("[preflight] auth/verify:", json.dumps(team, ensure_ascii=False))
    print("[preflight] budget/check:", json.dumps(budget, ensure_ascii=False))
    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.mode == "run" and not args.i_understand_run_is_scored:
        print("Refusing --mode run without --i_understand_run_is_scored "
              "(protects the 6 official scored runs).", file=sys.stderr)
        return 2

    if args.preflight:
        return preflight(args)

    # Full driver loop lives in driver.py (Phase 3+).
    from .driver import run_driver
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
