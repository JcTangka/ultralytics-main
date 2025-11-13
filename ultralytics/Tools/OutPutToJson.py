from ultralytics import YOLO
import json
import os

# 1. 加载训练好的模型
model = YOLO("D://Work//PYworks//ultralytics-main//runs//detect//train8//weights//best.pt")

# 2. 推理
results = model.predict("D://Work//PYworks//ultralytics-main//ultralytics//datasets//all_divide_my_dataset//train//images//day_people_image15.jpg", save=False, conf=0.25)

# 3. 构造JSON结果
output = []

for result in results:
    image_name = os.path.basename(result.path)
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

        detections.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": xyxy
        })

    output.append({
        "image_id": image_name,
        "detections": detections
    })

# 4. 保存为 JSON 文件
with open("detections.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
