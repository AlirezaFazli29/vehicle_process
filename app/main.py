from fastapi import (
    FastAPI,
    File, 
    UploadFile, 
    Form, 
    HTTPException,
)
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)
from contextlib import asynccontextmanager
from ultralytics import YOLO
from PIL import Image
import base64
import uvicorn


my_models = {}

@asynccontextmanager
async def lifspan(app: FastAPI):
    #####
    # models 
    #####
    yield
    my_models.clear()


app = FastAPI(
    title="vehicle process",
    lifespan=lifspan,
)


@app.get(
    path="/",
    tags=[
        "Vehicle Process",
        "Model Selection",
    ]
)
async def root():
    """
    Root endpoint for the service.

    This endpoint serves as a basic health check to verify if the service is up and running.
    When accessed, it returns a simple message indicating that the service is operational.

    Returns:
        JSONResponse: A JSON response with a message confirming the service status.
    """
    return JSONResponse(
        {"message": "Service is up and running"}
    )



uvicorn.run(app, host="0.0.0.0", port=8080)