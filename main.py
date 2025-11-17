from fastapi import FastAPI
import uvicorn
import os

from utils.logger import setup_logger
from dotenv import load_dotenv

from api import health
from api import documents
from api import query

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file = os.getenv("LOG_FILE")  # Optional
use_json = os.getenv("USE_JSON_LOGGING", "false").lower() == "true"

setup_logger(
    name="rag_system",
    level=log_level,
    log_file=log_file,
    use_json=use_json
)

app = FastAPI()


app.include_router(health.health_ep, prefix="/api")
app.include_router(documents.documents_ep, prefix="/api")
app.include_router(query.query_ep, prefix="/api")


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
