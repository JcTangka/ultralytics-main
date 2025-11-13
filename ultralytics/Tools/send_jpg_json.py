# -*- coding: utf-8 -*-
"""
detect_sender.py
发送端：实时检测监控流；当某帧检测到目标时，把“带有bounding box的这一帧图像(JPEG)”
和“该帧的JSON元数据”通过HTTP POST发送到指定服务器。

依赖安装：
    pip install ultralytics opencv-python requests

示例：
    python D://Work//PYworks//ultralytics-main//ultralytics//Tools//send_jpg_json.py --model D://Work//PYworks//ultralytics-main//runs//detect//train8//weights//best.pt
        --source "rtsp://admin:zz635241@192.168.0.127/Streaming/Channels/101"
        --server-url http://192.168.0.128:8081/api
        --conf 0.5 --cooldown 3

按 'q' 退出预览窗口；添加 --no-show 关闭本地窗口。
"""

import argparse
import json
import time
from datetime import datetime

import cv2
import requests
from ultralytics import YOLO


def now_ts_str():
    """形如 2025-09-13_14-05-30-123 的时间戳（毫秒精度）"""
    t = datetime.now()
    return t.strftime("%Y-%m-%d_%H-%M-%S-") + f"{int(t.microsecond/1000):03d}"


def encode_jpeg_bgr(image_bgr, quality=90) -> bytes:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, quality)))]
    ok, buf = cv2.imencode(".jpg", image_bgr, params)
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG.")
    return buf.tobytes()


def yolo_result_to_json(result, model_names):
    """
    将 YOLOv8 单帧结果转为 JSON 可序列化的字典：
    - timestamp
    - image: width, height
    - objects: [ {class_id, label, confidence, bbox_xyxy} ... ]
    """
    h, w = result.orig_shape if hasattr(result, "orig_shape") else (None, None)
    data = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "image": {"width": w, "height": h},
        "objects": []
    }

    if result.boxes is None or len(result.boxes) == 0:
        return data

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().tolist()
    conf = boxes.conf.cpu().tolist()
    cls = boxes.cls.cpu().tolist()

    for i in range(len(xyxy)):
        cls_id = int(cls[i])
        label = model_names.get(cls_id, str(cls_id))
        data["objects"].append({
            "class_id": cls_id,
            "label": label,
            "confidence": float(conf[i]),
            "bbox_xyxy": [float(v) for v in xyxy[i]]  # [x1,y1,x2,y2]
        })
    return data


def open_cv_source(source_str):
    """支持摄像头索引/RTSP/文件路径。"""
    try:
        if source_str.isdigit():
            return cv2.VideoCapture(int(source_str))
    except Exception:
        pass
    return cv2.VideoCapture(source_str)


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 实时检测发送端（仅发送端）")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 权重路径或名称")
    ap.add_argument("--source", type=str, default="0", help="视频源：摄像头索引(如0)/RTSP/文件路径")
    ap.add_argument("--server-url", type=str, required=True, help="接收端完整URL，如 http://IP:PORT/upload")
    ap.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值")
    ap.add_argument("--cooldown", type=float, default=0.5, help="最短发送间隔(秒)，避免过于频繁")
    ap.add_argument("--jpeg-quality", type=int, default=90, help="JPEG质量 1-100")
    ap.add_argument("--timeout", type=float, default=10.0, help="HTTP超时(秒)")
    ap.add_argument("--no-show", action="store_true", help="不显示本地可视化窗口")
    args = ap.parse_args()

    print(f"[sender] loading model: {args.model}")
    model = YOLO(args.model)

    cap = open_cv_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"[sender] ERROR: cannot open source: {args.source}")
    print(f"[sender] streaming from: {args.source}")
    print(f"[sender] posting to: {args.server_url}")
    print(f"[sender] conf={args.conf}, iou={args.iou}, cooldown={args.cooldown}s")

    session = requests.Session()
    last_sent_ts = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # 某些 RTSP 会偶发读失败，短暂等待后继续
                time.sleep(0.03)
                continue

            # 推理（逐帧）
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )
            result = results[0]

            # 画框预览图
            vis_img = result.plot()

            # 当有目标且过了冷却时间才发送
            has_obj = (result.boxes is not None) and (len(result.boxes) > 0)
            now = time.time()
            if has_obj and (now - last_sent_ts) >= float(args.cooldown):
                last_sent_ts = now

                # JSON 元数据
                meta = yolo_result_to_json(result, model.names)

                # 编码 JPG
                jpg_bytes = encode_jpeg_bgr(vis_img, quality=args.jpeg_quality)

                # 发送
                ts = now_ts_str()
                filename = f"frame_{ts}.jpg"
                files = {
                    "image": (filename, jpg_bytes, "image/jpeg"),
                }
                data = {
                    "meta": json.dumps(meta, ensure_ascii=False)
                }

                try:
                    resp = session.post(args.server_url, files=files, data=data, timeout=args.timeout)
                    if resp.status_code == 200:
                        print(f"[sender] sent {filename} | objs={len(meta['objects'])}")
                    else:
                        print(f"[sender] FAIL {resp.status_code}: {resp.text[:200]}")
                except Exception as e:
                    print(f"[sender] ERROR posting: {e}")

            # 本地显示
            if not args.no_show:
                cv2.imshow("YOLOv8 Live (Sender)", vis_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
