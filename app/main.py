from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.config.settings import settings

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.project_name}"}


app.include_router(
    health_router,
    prefix=settings.api_prefix,
)
