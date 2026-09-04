"""Local web UI — a FastAPI app over the same pipeline.

    stocks-web            # or: uvicorn stocks.web.app:app --reload
    open http://127.0.0.1:8000
"""

from stocks.web.app import app, main

__all__ = ["app", "main"]
