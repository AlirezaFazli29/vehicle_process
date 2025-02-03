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
from utils import (
    process_yolo_result,
    YoloJSONRequest,
    ModelJSONRequest,
)
from model_handler import YoloType
from PIL import Image
import base64
import io
import uvicorn


my_models = {}

@asynccontextmanager
async def lifspan(app: FastAPI):
    my_models["yolo_plate"] = YOLO(YoloType.CustomPlate.Plate_best.value)
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


@app.post(
        path="/file-to-base64",
        tags=["Vehicle Process"],
)
async def file_to_base64(file: UploadFile):
    """
    Convert an uploaded file to a base64-encoded string.

    This function processes an uploaded file and converts its binary content
    into a base64-encoded string. The resulting string can be used for
    serialization or transmission of the file content in text format.
    If any error occurs during file processing, it raises an appropriate 
    HTTP exception with a message.

    Args:
        file (UploadFile): The file uploaded by the client.

    Returns:
        JSONResponse: A JSON response containing the filename and the 
                      base64-encoded string of the file content.
    """
    try:
        file_data = await file.read()
        base64_string = base64.b64encode(file_data).decode('utf-8')
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process the uploaded file.")

    return JSONResponse(
        content={"filename": file.filename, "base64_string": base64_string}
    )


@app.post(
        path="/find-plate-bb",
        tags=["Vehicle Process"],
)
async def find_plate_bb(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
    return_base64_result: bool = Form(...),
):
    """
    Detects license plates in the given image using a YOLO model.

    Args:
        file (UploadFile): The uploaded image file for processing.
        conf_threshold (float): Confidence threshold for plate detection.
        return_base64_result (bool): Whether to return the processed image as base64.

    Returns:
        JSONResponse: A response containing detected plate bounding boxes,
                      the original image dimensions, and optionally the processed image in base64.
    """
    image = Image.open(file.file)
    res = my_models["yolo_plate"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    response = {
        "data": response,
        "origin_image_size": {
            "x": res[0].orig_shape[1],
            "y": res[0].orig_shape[0],
        }
    }
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_result_image"] = base64_image
    return JSONResponse(response)


@app.post(
        path="/find-plate-bb-plot",
        tags=["Vehicle Process"],
)
async def find_plate_bb_plot(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Detects license plates and returns a plotted image with bounding boxes.

    Args:
        file (UploadFile): The uploaded image file for processing.
        conf_threshold (float): Confidence threshold for plate detection.

    Returns:
        StreamingResponse: A streamed response with the processed image containing detected plates.
    """
    image = Image.open(file.file)
    res = my_models["yolo_plate"](image, conf=conf_threshold, verbose=False)
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post(
        path="/find-plate-bb-base64-input",
        tags=["Vehicle Process"],
)
async def find_plate_bb_base64(
    request: YoloJSONRequest
):
    """
    Detects license plates from a base64-encoded image input.

    Args:
        request (YoloJSONRequest): A JSON request containing the base64-encoded image,
                                   confidence threshold, and whether to return base64 output.

    Returns:
        JSONResponse: A response containing detected plate bounding boxes,
                      the original image dimensions, and optionally the processed image in base64.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    return_base64_result = request.return_base64_result
    res = my_models["yolo_plate"](image, conf=conf_threshold, verbose=False)
    response = process_yolo_result(res[0])
    response = {
        "data": response,
        "origin_image_size": {
            "x": res[0].orig_shape[1],
            "y": res[0].orig_shape[0],
        }
    }
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_result_image"] = base64_image
    return JSONResponse(response)


@app.get(
        path="/show_model_types",
        tags=["Model Selection"],
)
async def show_model_types():
    """"""
    yolo_plate_types = {
        model_type.name: model_type.value for model_type in YoloType.CustomPlate
    }
    yolo_truck_types = {
        model_type.name: model_type.value for model_type in YoloType.CustomTruck
    }
    yolo_plate_ocr_types = {
        model_type.name: model_type.value for model_type in YoloType.CustomPlateOCR
    }
    return JSONResponse(
        {
            "Find plate bounding boxes model": yolo_plate_types,
            "Truck type detection model": yolo_truck_types,
            "Plate OCR model": yolo_plate_ocr_types,
        }
    )


@app.post(
    path="/select-bb-plate-model",
    tags=["Model Selection"],
)
async def select_bb_plate_model(
    model_type: YoloType.CustomPlate = Form(...),
):
    """
    Endpoint to select a YOLO model for plate bounding box detection.

    This function allows the user to choose a specific model for detecting plate bounding boxes.
    The selected model is then stored in a global variable for use in future processing.

    Args:
        model_type (YoloType.CustomPlate): The model type to be selected for plate bounding box detection.

    Returns:
        JSONResponse: A JSON response confirming the model selection.
    """
    global my_models
    my_models["yolo_plate"] = YOLO(model_type.value)
    return JSONResponse(
        {"message": f"Model {model_type.name} is selected"}
    )


@app.post(
    path="/select-bb-plate-model-base64-input",
    tags=["Model Selection"],
)
async def select_bb_plate_model_base64(
    request: ModelJSONRequest,
):
    """
    Endpoint to select a YOLO model for plate bounding box detection from a base64-encoded input.

    This function allows the user to choose a YOLO model via a request containing base64-encoded model data.
    The model path or type specified in the request is used to load the appropriate YOLO model.

    Args:
        request (ModelJSONRequest): The request containing the model path or type in base64 format.

    Returns:
        JSONResponse: A JSON response confirming the model selection from the base64 input.
    """
    global my_models
    my_models["yolo_plate"] = YOLO(request.model_type)
    return JSONResponse(
        {"message": f"Model from path {request.model_type} is selected"}
    )


uvicorn.run(app, host="0.0.0.0", port=8080)