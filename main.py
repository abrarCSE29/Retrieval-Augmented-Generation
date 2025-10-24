from fastapi import FastAPI
import uvicorn

from api import health


app = FastAPI()


app.include_router(health.health_ep, prefix="/api")


# if __name__ == "__main__":
#     uvicorn.run("main:app", reload=True)