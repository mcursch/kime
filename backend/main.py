from fastapi import FastAPI

app = FastAPI(title="Kime API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Liveness check — returns HTTP 200 when the server is up."""
    return {"status": "ok"}
