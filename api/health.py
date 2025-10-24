from fastapi import APIRouter

health_ep = APIRouter(prefix="/health",tags=["health"])

@health_ep.get("")
def get_health():
    return {
        "status" : "success",
        "message" : "RAG API endpoint Active"
    }