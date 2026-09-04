"""Stock evaluation system — deterministic core with a thin LLM layer.

See ARCHITECTURE.md for the design. The short version:

    resolve -> deterministic fetch/compute -> 3 LLM agents -> compose
    -> deterministic scoring -> deterministic QA -> render

The LLM touches language, never arithmetic.
"""

__version__ = "0.2.0"
