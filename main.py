"""Root entrypoint for production (Railpack / Railway / etc.).
Runs the FastAPI app from api.main with uvicorn."""
import os
import uvicorn
from api.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
