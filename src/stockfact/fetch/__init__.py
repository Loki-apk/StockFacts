"""The deterministic fetch + compute layer, step 2 in ARCHITECTURE.md.

No LLM anywhere in here. Each module below takes raw provider data and turns
it into one typed report, then writes it onto the ``RunContext``. They don't
depend on each other (mostly), so ``pipeline.py`` runs them in parallel.
"""

from stockfact.fetch.pipeline import run_fetch_layer

__all__ = ["run_fetch_layer"]
