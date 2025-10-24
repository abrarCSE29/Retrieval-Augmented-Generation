from fastapi import APIRouter

from utils.api_response_format import create_api_response

health_ep = APIRouter(prefix="/health",tags=["health"])

@health_ep.get("")
def get_health():
    
    return create_api_response(
        status="success",
        message="RAG AI API WORKING",
    )