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
    select_device,
    process_yolo_result,
    process_yolo_result_ocr,
    rectify,
    YoloJSONRequest,
    YoloJSONRequest2,
    ModelJSONRequest,
    UnetJSONRequest,
)
from my_models import (
    Plate_Unet,
)
from torchvision import transforms
from model_handler import (
    YoloType,
    UNetType
)
from PIL import Image
import numpy as np
import base64
import io
import uvicorn


my_models = {}
my_transforms = {}

@asynccontextmanager
async def lifspan(app: FastAPI):
    global device
    device = select_device()
    my_models["yolo_plate"] = YOLO(YoloType.CustomPlate.Plate_best.value)
    my_models["plate_unet"] = Plate_Unet(UNetType.Corner_best, device=device)
    my_models["yolo_ocr"] = m = YOLO(YoloType.CustomPlateOCR.plate_ocr_best.value)
    my_transforms["unet"] = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            )
        ]
    )
    yield
    my_models.clear()
    my_transforms.clear()


app = FastAPI(
    title="Plate Bounding Box",
    lifespan=lifspan,
)


@app.get(
    path="/",
    tags=[
        "Plate Bounding Box",
        "Plate Rectification",
        "Plate OCR",
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
        tags=[
            "Plate Bounding Box",
            "Plate Rectification",
            "Plate OCR",
        ]
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
        tags=["Plate Bounding Box"],
)
async def find_plate_bb(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
    return_base64_result: bool = Form(...),
    return_base64_cropped_plates: bool = Form(...),
):
    """
    Detects license plates in the given image using a YOLO model and returns relevant information.

    Args:
        file (UploadFile): The uploaded image file to process.
        conf_threshold (float): Confidence threshold for plate detection. Plates 
                                with lower confidence will be ignored.
        return_base64_result (bool): Flag to indicate whether to return the processed 
                                     image as a base64 string.
        return_base64_cropped_plates (bool): Flag to indicate whether to return the 
                                             cropped plates as base64-encoded images.

    Returns:
        JSONResponse: A response containing:
            - `data`: Detected plate bounding boxes and related information.
            - `origin_image_size`: The original dimensions of the uploaded image.
            - Optionally, `base64_result_image` containing the processed image as base64 
              if `return_base64_result` is True.
            - Optionally, `base64_cropped_plates` containing base64-encoded cropped plate 
              images if `return_base64_cropped_plates` is True.
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
    if return_base64_cropped_plates:
        boxes = response["data"][0]["boxes"]
        plates = [None] * len(boxes)
        for i, box in enumerate(boxes):
            _, x1, y1, x2, y2 = map(int, box.values())
            plates[i] = np.array(image)[y1:y2, x1:x2, :]
            plates[i] = Image.fromarray(plates[i])
            buffer = io.BytesIO()
            plates[i].save(buffer, format="PNG")
            buffer.seek(0)
            plates[i] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_cropped_plates"] = plates
    return JSONResponse(response)


@app.post(
        path="/find-plate-bb-plot",
        tags=["Plate Bounding Box"],
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
        tags=["Plate Bounding Box"],
)
async def find_plate_bb_base64(
    request: YoloJSONRequest
):
    """
    Detects license plates from a base64-encoded image input and returns relevant information.

    Args:
        request (YoloJSONRequest): A JSON object containing the following fields:
            - `base64_string`: The base64-encoded image string to be processed.
            - `conf_threshold`: Confidence threshold for plate detection. Plates with 
                                lower confidence will be ignored.
            - `return_base64_result`: Flag indicating whether to return the processed 
                                      image as base64.
            - `return_base64_cropped_plates`: Flag indicating whether to return the 
                                              cropped plates as base64-encoded images.

    Returns:
        JSONResponse: A response containing:
            - `data`: Detected plate bounding boxes along with related information.
            - `origin_image_size`: The original dimensions of the input image.
            - Optionally, `base64_result_image`: The processed image in base64 format, 
                                                 if `return_base64_result` is True.
            - Optionally, `base64_cropped_plates`: A list of cropped plate images encoded 
                                                   in base64 format, if `return_base64_cropped_plates` 
                                                   is True.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    return_base64_result = request.return_base64_result
    return_base64_cropped_plates = request.return_base64_cropped_plates
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
    if return_base64_cropped_plates:
        boxes = response["data"][0]["boxes"]
        plates = [None] * len(boxes)
        for i, box in enumerate(boxes):
            _, x1, y1, x2, y2 = map(int, box.values())
            plates[i] = np.array(image)[y1:y2, x1:x2, :]
            plates[i] = Image.fromarray(plates[i])
            buffer = io.BytesIO()
            plates[i].save(buffer, format="PNG")
            buffer.seek(0)
            plates[i] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_cropped_plates"] = plates
    return JSONResponse(response)


@app.post(
        path="/rectify-plate-plot",
        tags=["Plate Rectification"]
)
async def rectify_plate_plot(
    file: UploadFile = File(...),
):
    """
    Processes an uploaded image, segments the plate using a UNet model, 
    rectifies it, and returns the transformed image.

    Args:
        file (UploadFile): The uploaded image file for processing.

    Returns:
        StreamingResponse: A streamed response with the rectified plate image.
    """
    image = Image.open(file.file)
    image_torch = my_transforms["unet"](image)
    image_torch = image_torch.unsqueeze(0).to(device)
    plate_segment = my_models["plate_unet"](image_torch)
    segment_array = plate_segment[0].squeeze(0).cpu().numpy()
    rectified_image = rectify(
        image = image,
        segmentation_result = segment_array,
    )
    result_pil = Image.fromarray(rectified_image)
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post(
        path="/rectify-plate-base64-input",
        tags=["Plate Rectification"]
)
async def rectify_plate_base64(
    request: UnetJSONRequest,
):
    """
    Processes a base64-encoded image, segments the plate using a UNet model, 
    rectifies it, and returns the transformed image in base64 format.

    Args:
        request (UnetJSONRequest): The request containing a base64-encoded image string.

    Returns:
        JSONResponse: A JSON response containing the base64 string of the rectified plate image.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    image_torch = my_transforms["unet"](image)
    image_torch = image_torch.unsqueeze(0).to(device)
    plate_segment = my_models["plate_unet"](image_torch)
    segment_array = plate_segment[0].squeeze(0).cpu().numpy()
    rectified_image = rectify(
        image = image,
        segmentation_result = segment_array,
    )
    result_pil = Image.fromarray(rectified_image)
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return JSONResponse(
        {"base64_rectified_plate": base64_image}
    )


@app.post(
        path="/ocr-plate",
        tags=["Plate OCR"],
)
async def ocr_plate(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
    return_base64_result: bool = Form(...),
):
    """
    Endpoint to perform OCR on a license plate image.

    Parameters:
    - file (UploadFile): The uploaded image file.
    - conf_threshold (float): Confidence threshold for YOLO OCR detection.
    - return_base64_result (bool): If True, returns the processed image as a base64 string.

    Returns:
    - JSONResponse: A dictionary containing the extracted OCR text.
                    If requested, includes the processed image as a base64 string.
    """
    image = Image.open(file.file)
    res = my_models["yolo_ocr"](image, conf=conf_threshold, verbose=False)
    ocr_result = process_yolo_result_ocr(res[0])
    response = {"ocr-result": ocr_result}
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_result_image"] = base64_image
    return JSONResponse(response)


@app.post(
        path="/ocr-plate-base64-input",
        tags=["Plate OCR"],
)
async def ocr_plate_base64(
    request: YoloJSONRequest2,
):
    """
    Endpoint to perform OCR on a license plate image provided as a base64 string.

    Parameters:
    - request (YoloJSONRequest2): JSON payload containing:
        - base64_string (str): The base64-encoded image data.
        - conf_threshold (float): Confidence threshold for YOLO OCR detection.
        - return_base64_result (bool): If True, returns the processed image as a base64 string.

    Returns:
    - JSONResponse: A dictionary containing the extracted OCR text.
                    If requested, includes the processed image as a base64 string.
    """
    image_data = base64.b64decode(request.base64_string)
    image = Image.open(io.BytesIO(image_data))
    conf_threshold = request.conf_threshold
    return_base64_result = request.return_base64_result
    res = my_models["yolo_ocr"](image, conf=conf_threshold, verbose=False)
    ocr_result = process_yolo_result_ocr(res[0])
    response = {"ocr-result": ocr_result}
    if return_base64_result:
        result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        response["base64_result_image"] = base64_image
    return JSONResponse(response)


@app.post(
        path="/ocr-plate-plot",
        tags=["Plate OCR"],
)
async def ocr_plate_plot(
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    """
    Endpoint to perform Optical Character Recognition (OCR) on a license plate image.

    This function processes an uploaded image, applies a YOLO-based OCR model,
    and returns the annotated image with detected characters.

    Args:
        file (UploadFile): The uploaded image file for OCR processing.
        conf_threshold (float): Confidence threshold for OCR detection.

    Returns:
        StreamingResponse: A streamed response containing the annotated image.
    """
    image = Image.open(file.file)
    res = my_models["yolo_ocr"](image, conf=conf_threshold, verbose=False)
    result_pil = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


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
    unet_plate_rectification = {
        model_type.name: model_type.value for model_type in UNetType
    }
    return JSONResponse(
        {
            "Find plate bounding boxes model": yolo_plate_types,
            "Truck type detection model": yolo_truck_types,
            "Plate OCR model": yolo_plate_ocr_types,
            "Plate rectification model": unet_plate_rectification,
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


@app.post(
    path="/select-plate-rectification-model",
    tags=["Model Selection"],
)
async def select_plate_rectification_model(
    model_type: UNetType = Form(...),
):
    """
    Endpoint to select a UNet model for plate rectification.

    This function allows the user to select a specific UNet model type
    for plate rectification by providing a model type as form input.

    Args:
        model_type (UNetType): The selected UNet model type.

    Returns:
        JSONResponse: A JSON response confirming the model selection.
    """
    global my_models
    my_models["plate_unet"] = Plate_Unet(UNetType.Corner_best, device=device)
    return JSONResponse(
        {"message": f"Model {model_type.name} is selected"}
    )


@app.post(
    path="/select-plate-rectification-base64-input",
    tags=["Model Selection"],
)

async def select_plate_rectification_model_base64(
    request: ModelJSONRequest,
):
    """
    Endpoint to select a UNet model for plate bounding box detection.

    This function allows the user to choose a UNet model via a request containing 
    model path or type in base64-encoded format.
    The specified model is then loaded for processing.

    Args:
        request (ModelJSONRequest): The request containing the base64-encoded model path or type.

    Returns:
        JSONResponse: A JSON response confirming the model selection.
    """
    global my_models
    my_models["plate_unet"] = Plate_Unet(UNetType.Corner_best, device=device)
    return JSONResponse(
        {"message": f"Model from path {request.model_type} is selected"}
    )



uvicorn.run(app, host="0.0.0.0", port=8080)