import uvicorn
from app_core.api.v1.router import api_router
from fastapi import FastAPI

app = FastAPI(title="HouTrellis Modular API Service", version="1.0.0")

# Register V1 Router under /api/v1 prefix
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "HouTrellis API is fully active and running on RTX 4090!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
