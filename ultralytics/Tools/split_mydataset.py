import os
import shutil
import random


def split_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, random_seed=42):
    """
    参数说明：
    - images_dir: 原始图片目录路径
    - labels_dir: 原始标签目录路径
    - output_dir: 输出根目录路径
    - train_ratio: 训练集比例（默认0.8）
    - random_seed: 随机种子（确保可重复性）
    """
    # 设置随机种子
    random.seed(random_seed)

    # 创建目标目录结构
    dirs = [
        os.path.join(output_dir, 'train/images'),
        os.path.join(output_dir, 'train/labels'),
        os.path.join(output_dir, 'val/images'),
        os.path.join(output_dir, 'val/labels')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 验证文件对应关系
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

    # 去除扩展名比较
    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    label_basenames = {os.path.splitext(f)[0] for f in label_files}

    # 检查不匹配项
    missing_labels = image_basenames - label_basenames
    extra_labels = label_basenames - image_basenames

    if missing_labels:
        print(f"警告：发现 {len(missing_labels)} 个图片没有对应标签")
    if extra_labels:
        print(f"警告：发现 {len(extra_labels)} 个标签没有对应图片")

    # 获取有效文件对
    valid_pairs = [(f, os.path.splitext(f)[0] + '.txt') for f in image_files
                   if os.path.splitext(f)[0] in label_basenames]

    # 随机打乱顺序
    random.shuffle(valid_pairs)

    # 计算分割点
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    # 复制文件的函数
    def copy_files(pairs, phase):
        for img_file, label_file in pairs:
            # 复制图片
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, phase, 'images', img_file)
            shutil.copy(src_img, dst_img)

            # 复制标签
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, phase, 'labels', label_file)
            shutil.copy(src_label, dst_label)

    # 执行复制
    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')

    # 输出统计信息
    print(f"\n拆分完成！最终数据集结构：")
    print(f"总样本数：{len(valid_pairs)}")
    print(f"训练集：{len(train_pairs)} 个样本 ({len(train_pairs) / len(valid_pairs):.1%})")
    print(f"验证集：{len(val_pairs)} 个样本 ({len(val_pairs) / len(valid_pairs):.1%})")
    print(f"输出目录结构：")
    print(f"└── {output_dir}")
    print(f"    ├── train/")
    print(f"    │   ├── images/  # 训练图片")
    print(f"    │   └── labels/  # 训练标签")
    print(f"    └── val/")
    print(f"        ├── images/  # 验证图片")
    print(f"        └── labels/  # 验证标签")


# 使用示例
if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    original_images = "D:\\Work\\PYworks\\ultralytics-main\\ultralytics\\datasets\\images"  # 原始图片目录
    original_labels = "D:\\Work\\PYworks\\ultralytics-main\\ultralytics\\datasets\\labels"  # 原始标签目录
    output_directory = "D:\\Work\\PYworks\\ultralytics-main\\ultralytics\\datasets"  # 输出目录

    split_yolo_dataset(
        images_dir=original_images,
        labels_dir=original_labels,
        output_dir=output_directory,
        train_ratio=0.8,  # 8:2比例
        random_seed=42  # 固定随机种子确保可重复
    )