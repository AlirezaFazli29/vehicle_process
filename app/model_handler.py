from enum import Enum


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