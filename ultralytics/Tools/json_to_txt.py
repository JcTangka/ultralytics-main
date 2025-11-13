import json
import os
from pathlib import Path

# 配置项
json_path = "../datasets/SSLAD-2D/annotations/instance_val.json"  # JSON文件路径
output_dir = "../datasets/SSLAD-2D/val/labels"  # 输出目录
start_id = 1  # 起始image_id
end_id = 50  # 结束image_id

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取JSON文件
with open(json_path) as f:
    data = json.load(f)

# 构建图像信息映射表（仅处理1-200）
image_info = {
    img["id"]: {
        "file_name": Path(img["file_name"]).stem,
        "width": img["width"],
        "height": img["height"]
    }
    for img in data["images"]
    if start_id <= img["id"] <= end_id
}

# 创建类别映射
category_map = {cat["id"]: cat["name"] for cat in data["categories"]}

# 统计类别信息
print("检测到以下类别：")
for cat_id, cat_name in category_map.items():
    print(f"ID: {cat_id} -> {cat_name}")

# 按image_id分组标注（仅处理1-200）
annotations = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    if start_id <= img_id <= end_id:
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)

# 处理每个图像的标注
processed_count = 0
for img_id, anns in annotations.items():
    # 验证图像信息存在
    if img_id not in image_info:
        print(f"警告：跳过未找到尺寸信息的image_id {img_id}")
        continue

    # 获取图像信息
    info = image_info[img_id]
    filename = info["file_name"]
    img_w = info["width"]
    img_h = info["height"]

    # 生成输出路径
    txt_path = os.path.join(output_dir, f"{filename}.txt")

    with open(txt_path, "w") as f:
        for ann in anns:
            # 解析原始bbox [x, y, w, h]
            x, y, w, h = ann["bbox"]

            # 归一化计算
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # YOLO格式：class_id x_center y_center width height
            line = f"{ann['category_id'] - 1} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            f.write(line)

    processed_count += 1

print(f"\n转换完成！共处理 {processed_count}/{len(image_info)} 个有效图像（ID {start_id}-{end_id}）")