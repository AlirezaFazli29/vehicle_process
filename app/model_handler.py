from enum import Enum
from abc import ABC, abstractmethod


class YoloType():
    """Enumeration for model types."""

    class Pretrained(Enum):
        yolo8n = "yolo8n.pt"
        yolo8s = "yolo8s.pt"
        yolo8m = "yolo8m.pt"
        yolo8l = "yolo8l.pt"
        yolo8x = "yolo8x.pt"
        yolo9n = "yolo9n.pt"
        yolo9s = "yolo9s.pt"
        yolo9m = "yolo9m.pt"
        yolo9l = "yolo9l.pt"
        yolo9x = "yolo9x.pt"
        yolo10n = "yolo10n.pt"
        yolo10s = "yolo10s.pt"
        yolo10m = "yolo10m.pt"
        yolo10l = "yolo10l.pt"
        yolo10x = "yolo10x.pt"
        yolo11n = "yolo11n.pt"
        yolo11s = "yolo11s.pt"
        yolo11m = "yolo11m.pt"
        yolo11l = "yolo11l.pt"
        yolo11x = "yolo11x.pt"

    class CustomPlate(Enum):
        Plate_last = "weights/last(plate).pt"
        Plate_best = "weights/best(plate).pt"

    class CustomTruck(Enum):
        truck_best = "weights/truck(best).pt"
        truck_last = "weights/truck(last).pt"

    class CustomPlateOCR(Enum):
        plate_ocr_best = "weights/best(ocr).pt"
        plate_ocr_last = "weights/last(ocr).pt"


class UNetType(Enum):
    """Enumeration for model types."""
    Corner_last = 'weights/last_custom_u.pth'
    Corner_best = 'weights/last_custom_u.pth'
    Base = 'base'

    
class BaseModel(ABC):
    """
    Abstract base class for machine learning models.

    This class defines the required interface that all model subclasses must implement.
    It ensures that every model has methods for initialization, loading, and inference.
    """
    @abstractmethod
    def __init__(self):
        """
        Initializes the model.

        Subclasses must implement this method to define any necessary attributes,
        such as model configuration, parameters, or loading mechanisms.
        """
        pass  # To be implemented in subclasses

    @abstractmethod
    def load_model(self):
        """
        Loads the model.

        This method should be implemented by subclasses to handle loading model weights,
        architectures, or any preprocessing steps needed before inference.
        """
        pass  # To be implemented in subclasses

    @abstractmethod
    def __call__(self, image):
        """
        Runs inference on an input image.

        Args:
            image: The input data on which inference should be performed.
        
        Returns:
            The model’s output after processing the input image.
        """
        pass  # To be implemented in subclasses

    def __str__(self):
        """
        Returns a string representation of the model.

        This method assumes that subclasses define a `model_path` attribute.
        If `model_path` is not defined in a subclass, this method may need to be overridden.
        """
        return f"Model: {self.model_path}"