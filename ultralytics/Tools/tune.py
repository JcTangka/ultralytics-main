# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # 需要修改
    model.tune(
        data=r"../datasets/SSLAD-2D/my_SSLAD.yaml",  # 需要修改
        imgsz=640,
        epochs=100,
        iterations=50,
    )
