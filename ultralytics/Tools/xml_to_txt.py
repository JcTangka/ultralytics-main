import os
import xml.etree.ElementTree as ET
from glob import glob


def xml_to_yolo(xml_dir, txt_dir):
    # 确保输出目录存在
    os.makedirs(txt_dir, exist_ok=True)

    # 第一次遍历：收集所有类别
    classes = set()
    xml_files = glob(os.path.join(xml_dir, '*.xml'))

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip()
            classes.add(cls_name)

    # 按字母顺序排序类别
    sorted_classes = sorted(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}

    # 第二次遍历：处理文件
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 准备输出文件路径
        txt_name = os.path.splitext(os.path.basename(xml_file))[0] + '.txt'
        txt_path = os.path.join(txt_dir, txt_name)

        with open(txt_path, 'w') as f:
            for obj in root.iter('object'):
                # 获取类别索引
                cls_name = obj.find('name').text.strip()
                cls_idx = class_to_idx[cls_name]

                # 解析边界框
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # 归一化处理
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                # 写入文件
                f.write(f"{cls_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # 输出统计信息
    print(f"转换完成，共处理 {len(xml_files)} 个文件")
    print("\n标注数据包含以下类别（按字母顺序排序）：")
    for idx, cls in enumerate(sorted_classes):
        print(f"{idx}: {cls}")


# 使用示例
xml_dir = "D:\\Work\\PYworks\\ultralytics-main\\ultralytics\\datasets\\xmls"  # 替换为你的XML目录路径
txt_dir = "D:\\Work\\PYworks\\ultralytics-main\\ultralytics\\datasets\\labels"  # 替换为输出目录路径
xml_to_yolo(xml_dir, txt_dir)

