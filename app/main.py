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
from PIL import Image
import base64


my_models = {}

@asynccontextmanager
async def lifspan(app: FastAPI):
    #####
    # models 
    #####
    yield
    my_models.clear()