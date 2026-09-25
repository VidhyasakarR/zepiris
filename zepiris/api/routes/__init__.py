from fastapi import APIRouter

from zepiris.api.routes import checkpoint, dresscode, face, health, quality, ui


def build_api_router() -> APIRouter:
    root = APIRouter()
    root.include_router(health.router, tags=["health"])
    root.include_router(face.router, prefix="/v1/faces", tags=["faces"])
    root.include_router(dresscode.router, prefix="/v1/dresscode", tags=["dresscode"])
    root.include_router(checkpoint.router, prefix="/v1/checkpoint", tags=["checkpoint"])
    root.include_router(quality.router, prefix="/v1/quality", tags=["quality"])
    root.include_router(ui.router)
    return root
