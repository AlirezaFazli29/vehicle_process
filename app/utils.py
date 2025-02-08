from torch.cuda import (
    is_available,
    get_device_name
)
from ultralytics.engine.results import Results
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
import tqdm
import os
import cv2
import warnings
from sklearn.mixture import GaussianMixture
import joblib


def select_device():
    """
    Selects the appropriate device for computation based on GPU availability.

    This function checks if a CUDA-compatible GPU is available. 
    - If a GPU is detected, it prints a message indicating the GPU is selected.
    - If no GPU is found, it defaults to the CPU and prints a message accordingly.

    Returns:
        str: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if is_available(): 
        print(
            f"{get_device_name()} have been located and selected"
        )
    else: 
        print(
            "No GPU cuda core found on this device. cpu is selected as network processor"
        )
    return 'cuda' if is_available() else 'cpu'


def process_yolo_result(result: Results) -> list:
    """
    Process the YOLO inference result and group detections by object type.

    This function takes the result from a YOLO inference, which includes detected 
    objects with their respective bounding boxes and confidence scores, and then 
    groups these detections by object name (e.g., "person", "car", etc.). It returns 
    a list of dictionaries containing the object name, the count of detections for each object, 
    and the associated bounding boxes and confidence scores.

    Args:
        result (Results): The result from the YOLO model inference, typically containing 
                          a list of detected objects with their properties (name, 
                          confidence, bounding box coordinates).

    Returns:
        list: A list of dictionaries where each dictionary contains:
              - 'obj': The object name.
              - 'count': The number of detections for this object.
              - 'boxes': A list of bounding box data for each detection, including 
                confidence score and coordinates (x1, y1, x2, y2).
              If no objects are detected, a dictionary with an error message is returned.
    """
    summary = result.summary()
    if len(summary) > 0:
        grouped = {}
        for item in summary:
            obj_name = item['name']
            if obj_name not in grouped:
                grouped[obj_name] = {'class': obj_name, 'count': 0, 'boxes': []}
            grouped[obj_name]['boxes'].append({'conf': item['confidence'], **item['box']})
            grouped[obj_name]['count'] += 1
        return list(grouped.values())
    else: 
        return [{
            "error": 404,
            "message": "No object detected"
        }]


def get_gmm_features(
        center:np.ndarray, 
        n_components:int=3,
):
    """
    Compute Gaussian Mixture Model (GMM) features from center y-coordinates.

    Parameters:
    center (numpy.ndarray): A 1D array of y or x coordinates of the centers of bounding boxes.
    n_components (int): The number of Gaussian components for the GMM. Default is 3.

    Returns:
    numpy.ndarray: A feature vector containing concatenated values of:
        - Means of the GMM components.
        - Standard deviations (square root of covariances) of the GMM components.
        - Weights of the GMM components.
    """
    center = center.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(center)
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()
    std_devs = np.sqrt(covariances)
    return np.concatenate([means, std_devs, weights])


def process_yolo_result_ocr(result: Results) -> list:
    """
    Process YOLO OCR results to extract and sort detected text.

    Parameters:
    result (Results): The output of the YOLO model containing detected objects.

    Returns:
    list: A structured list containing sorted OCR results. If no objects are detected,
          returns an error message.
    """
    summary = result.summary()
    if len(summary) > 0:
        image_shape = result.orig_shape
        labels = np.array(
            [item["name"] for item in summary]
        )
        data = np.array(
            [
                [
                    (item["box"]["x2"]+item["box"]["x1"])/2,
                    (item["box"]["y2"]+item["box"]["y1"])/2,
                ] for item in summary
            ]
        )
        feature = get_gmm_features(
            data[:,1]/image_shape[0],
            n_components = 3,
        )
        svc_classifier = joblib.load("weights/svc_grid_model.pkl")
        plate_type = svc_classifier.predict(
            feature.reshape((1,len(feature)))
        ).item()
        datum =  np.column_stack([data, np.array(labels)])
        if plate_type <= 1:
            sorted_datum = [
                datum[datum[:,0].astype(np.float16).argsort()]
            ]
        else:
            gmm = GaussianMixture(n_components=plate_type)
            gmm.fit(
                datum[:,1].astype(np.float16).reshape(-1,1)
            )
            clusters = gmm.predict(
                datum[:,1].astype(np.float16).reshape(-1,1)
            )
            clustered_datum = [
                datum[clusters==cluster] for cluster in np.unique(clusters)
            ]
            sorted_datum = [
                clustered_datum[j][clustered_datum[j][:,0].astype(np.float16).argsort()]
                for j in gmm.means_.flatten().argsort()
            ]
        ocr_result = [
            [str(item) for item in line[:, -1]] for line in sorted_datum
        ]
        return ocr_result
    else: 
        return [{
            "error": 404,
            "message": "No object detected"
        }]


class PlatePipelineRequest(BaseModel):
    base64_string: str
    conf_threshold_bb: float = 0.7
    conf_threshold_ocr: float = 0.7
    return_plates_locations: bool = False
    return_base64_cropped_plates: bool = False
    return_base64_rectified_plates: bool = False
    return_base64_ocr_plate_results: bool = False

class YoloJSONRequest(BaseModel):
    base64_string: str
    conf_threshold: float = 0.7
    return_base64_result: bool = True
    return_base64_cropped_plates: bool = True


class YoloJSONRequest2(BaseModel):
    base64_string: str
    conf_threshold: float = 0.7
    return_base64_result: bool = True


class UnetJSONRequest(BaseModel):
    base64_string: str


class ModelJSONRequest(BaseModel):
    model_type: str


def train_custom_unet(
        model,
        train_loader,
        valid_loader,
        optimizer=None,
        lr:float=0.01,
        weight_decay=0.01,
        epochs:int=10,
        loss_fn=torch.nn.BCELoss(),
        device='cpu',
        save_parameters=True,
        save_path:str="./weights"
    ) -> dict:
    """
    Train a custom UNet model with specified parameters.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    optimizer (torch.optim.Optimizer, optional): Optimizer for model parameters. Defaults to Adam with lr.
    lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
    weight_decay (float, optional): Weight decay (L2 regularization) parameter for the optimizer. Defaults to 0.01.
    epochs (int, optional): Number of training epochs. Defaults to 10.
    loss_fn (torch.nn.Module, optional): Loss function used for training. Defaults to MSELoss.
    device (str, optional): Device for model training (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
    save_parameters (bool, optional): Flag to save model parameters. Defaults to True.
    save_path (str, optional): Path to save model weights. Defaults to './weights'.
    
    Returns:
    dict: A dictionary containing training and validation loss and training and validation IoU history.
    """
    model.to(device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'valid_loss': [], 'train_iou': [], 'valid_iou': []}
    min_loss_v = 1e9

    
    for ep in range(epochs):

        # Training Phase
        model.train()

        steps = train_loader.__len__()
        total_loss, total_iou, count = 0, 0, 0

        progress_bar = tqdm(train_loader, total=steps, desc= f'Training Epoch {ep+1:2}', leave=False)
        for feature, labels in progress_bar:
            imgs, lbls = feature.to(device), labels.to(device)
            optimizer.zero_grad()   
            out = model(imgs)   
            loss = loss_fn(out, lbls.unsqueeze(1)) 
            loss.backward()   
            optimizer.step()
            total_loss += loss
            iou = compute_iou(out, lbls.unsqueeze(1))
            total_iou += iou * len(lbls)  # Accumulate IoU
            count += len(lbls)
            progress_bar.set_postfix_str(f"Running Loss = {loss.item():.4f}, IoU = {iou.item():.4f}")
            progress_bar.update()

        # Calculate average training loss and IoU
        training_loss = total_loss.item() / count
        training_iou = total_iou.item() / count
        history["train_loss"].append(training_loss)
        history["train_iou"].append(training_iou)

        # Validation Phase
        model.eval()

        steps = valid_loader.__len__()
        total_loss, total_iou, count = 0, 0, 0

        with torch.no_grad():
            progress_bar = tqdm(valid_loader, total=steps, desc= f'Validating Epoch {ep+1:2}', leave=False)
            for feature, labels in progress_bar:         
                imgs, lbls = feature.to(device), labels.to(device)
                out = model(imgs)
                loss = loss = loss_fn(out, lbls.unsqueeze(1))  
                total_loss += loss
                iou = compute_iou(out, lbls.unsqueeze(1))
                total_iou += iou * len(lbls)  # Accumulate IoU
                count += len(lbls)
                progress_bar.set_postfix_str(f"Running Loss = {loss.item():.4f}, IoU = {iou.item():.4f}")
                progress_bar.update()

        # Calculate average validation loss and IoU
        validation_loss = total_loss.item() / count
        validation_iou = total_iou.item() / count
        history["valid_loss"].append(validation_loss)
        history["valid_iou"].append(validation_iou)
        
        if save_parameters: torch.save(model.state_dict(), os.path.join(save_path, "last_custom_u.pth"))
        if validation_loss<=min_loss_v:
            min_loss_v = validation_loss
            min_loss_t = training_loss
            if save_parameters: torch.save(model.state_dict(), os.path.join(save_path, "best_custom_u.pth"))

    print(f"Training Summary for {epochs} number of epochs:")
    print(f"    last epoch: Train loss = {training_loss:.6f}, train IoU = {training_iou:.6f}")
    print(f"                Valid loss = {validation_loss:.6f}, valid IoU = {validation_iou:.6f}")
    print(f"    best epoch: Train loss = {min_loss_t:.6f}, Valid loss = {min_loss_v:.6f}")

    return history


def compute_iou(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).float()
    intersection = (preds_binary * targets).sum()
    union = preds_binary.sum() + targets.sum() - intersection
    iou = intersection / (union + 1e-6)  # Adding a small value to avoid division by zero
    return iou


def rectify(
        image:Image.Image,
        segmentation_result:np.ndarray,
) -> np.ndarray:
    """
    Rectifies an image based on a given segmentation result.

    Args:
        image (Image.Image): The input image.
        segmentation_result (np.ndarray): The segmentation mask indicating the area to rectify.

    Returns:
        np.ndarray: The rectified image.
    """
    image_height, image_width = np.array(image).shape[:2]
    segmentation_result = cv2.resize(segmentation_result, (image_width, image_height))
    segmentation_result = (segmentation_result * 255).astype(np.uint8)
    _, binary = cv2.threshold(segmentation_result, 127, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        warnings.warn(
            "Could not find contours for this image. Returning original image."
        )
        return np.array(image)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    corners = None
    for i in np.arange(0.01, 0.055, 0.005):
        epsilon = i * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            corners = approx.squeeze(1)
            break
    if corners is None:
        warnings.warn(
            "Could not find corners for this image. Returning original image."
        )
        return np.array(image)
    sorted_corners = {
        "Top Left": list(
            corners[np.sqrt(np.sum((corners - np.array([0, 0])) ** 2, axis=1)).argmin()]
        ),
        "Top Right": list(
            corners[np.sqrt(np.sum((corners - np.array([image_width, 0])) ** 2, axis=1)).argmin()]
        ),
        "Bottom Right": list(
            corners[np.sqrt(np.sum((corners - np.array([image_width, image_height])) ** 2, axis=1)).argmin()]
        ),
        "Bottom Left": list(
            corners[np.sqrt(np.sum((corners - np.array([0, image_height])) ** 2, axis=1)).argmin()]
        ),
    }
    src_points = np.array(
        [
            sorted_corners["Top Left"], 
            sorted_corners["Top Right"], 
            sorted_corners["Bottom Right"], 
            sorted_corners["Bottom Left"],
        ],
        dtype = np.float32,
    )
    dst_points = np.array(
        [
            [0, 0], 
            [image_width, 0], 
            [image_width, image_height], 
            [0, image_height],
        ],
        dtype = np.float32,
    )
    M = cv2.getPerspectiveTransform(
        src_points.astype(np.float32),
        dst_points.astype(np.float32),
    )
    rectified_image = cv2.warpPerspective(
        np.array(image), M, (image_width, image_height)
    )
    return rectified_image


def get_gmm_features(
        center:np.ndarray, 
        n_components:int=3,
) -> np.ndarray:
    """
    Compute Gaussian Mixture Model (GMM) features from center y-coordinates.

    Parameters:
    center (numpy.ndarray): A 1D array of y or x coordinates of the centers of bounding boxes.
    n_components (int): The number of Gaussian components for the GMM. Default is 3.

    Returns:
    numpy.ndarray: A feature vector containing concatenated values of:
        - Means of the GMM components.
        - Standard deviations (square root of covariances) of the GMM components.
        - Weights of the GMM components.
    """
    center = center.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(center)
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()
    std_devs = np.sqrt(covariances)
    return np.concatenate([means, std_devs, weights])

