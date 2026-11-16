"""Offline routing simulator and scorecard (docs/03-routing.md#how-we-measure-routing). No model calls, no spend.

    uv run python scripts/routing_sim.py run                       # every strategy, the oracle, sensitivity: docs/routing-eval/
    uv run python scripts/routing_sim.py run --tasks 2000 --html   # a bigger run, with a static HTML report
    uv run python scripts/routing_sim.py run --plugin mypkg.v2:STRATEGIES   # score a strategy that isn't on main yet
    uv run python scripts/routing_sim.py check                     # the CI regression check against the stored baseline
    uv run python scripts/routing_sim.py check --update            # accept a change on purpose: write a new baseline
    uv run python scripts/routing_sim.py export --out rows/        # synthetic exported rows from an exploring run
    uv run python scripts/routing_sim.py replay rows/task_starts.jsonl --ledger rows/ledger.jsonl --policy one-down
"""

import sys

from meritorder.routing_eval import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
