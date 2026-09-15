"""A small local web UI — same pipeline underneath, just a FastAPI app on top.

    stockfact-web            # or: uvicorn stockfact.web.app:app --reload
    open http://127.0.0.1:8000
"""

from stockfact.web.app import app, main

__all__ = ["app", "main"]
