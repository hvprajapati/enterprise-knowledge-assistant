from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "Enterprise Knowledge Assistant",
    }
