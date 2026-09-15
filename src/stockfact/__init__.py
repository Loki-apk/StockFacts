"""StockFact — give it a ticker, get back a sourced report, not a prediction.

Full design notes are in ARCHITECTURE.md. Short version of how a run works:

    resolve -> deterministic fetch/compute -> 3 LLM agents -> compose
    -> deterministic scoring -> deterministic QA -> render

All the math happens in plain Python before the LLM ever sees the data. Its
only job is turning numbers that already exist into sentences.
"""

__version__ = "0.2.0"
