"""The three LLM agents (step 3 in ARCHITECTURE.md) — they write sentences, not numbers."""

from stockfact.agents.crew import run_llm_layer

__all__ = ["run_llm_layer"]
