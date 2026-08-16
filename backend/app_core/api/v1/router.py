from app_core.api.v1.endpoints import img23d, txt2img
from fastapi import APIRouter

api_router = APIRouter()

# Include endpoints under appropriate tags
api_router.include_router(txt2img.router, tags=["txt2img"])
api_router.include_router(img23d.router, tags=["img23d"])
