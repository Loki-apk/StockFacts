"""Deterministic fetch + compute layer (ARCHITECTURE.md step 2).

No LLM here. Each module turns provider data into one typed report and writes it
onto the ``RunContext``. Modules are independent and safe to run in parallel.
"""

from stocks.fetch.pipeline import run_fetch_layer

__all__ = ["run_fetch_layer"]
