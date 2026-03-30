"""Convert YOLOv8n.pt to ONNX format for browser inference."""
from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
model_path = SCRIPT_DIR / "yolov8n.pt"

model = YOLO(str(model_path))
model.export(format="onnx", imgsz=640, simplify=True, opset=13)
print(f"Exported to {SCRIPT_DIR / 'yolov8n.onnx'}")
