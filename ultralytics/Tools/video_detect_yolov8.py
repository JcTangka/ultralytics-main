# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# python D://Work//PYworks//ultralytics-main//ultralytics//Tools//video_detect_yolov8.py
# --model D://Work//PYworks//ultralytics-main//runs//detect//train8//weights//best.pt
# --source "rtsp://admin:tengfei888@192.168.1.64:554/Streaming/Channels/101"
# --source "rtsp://admin:zz635241@192.168.0.127/Streaming/Channels/101" //实验室监控 调试用
# --out_video D://Work//PYworks//ultralytics-main//runs//cam10//results//camera101_out.mp4
# --out_json D://Work//PYworks//ultralytics-main//runs//cam10//JSONLS//cam101.jsonl
# --imgsz 960 --conf 0.50 --device 0 --stride=5
# --show --view_scale 0.6 --rtmp_url rtmp://<192.168.0.128>/live/stream


# -*- coding: utf-8 -*-
"""
YOLOv8 实时/离线视频检测 + RTMP 推流（带框画面）
- 边跑边看（OpenCV 窗口）
- 保存带框视频（.mp4）
- 保存逐帧 JSON Lines（.jsonl），含时间戳
- RTMP 推流 annotated 帧到另一台电脑
- 统一输出目录：project/name/.

用法示例：
1) 推 RTMP + 本地显示 + 保存带框视频/JSON
python video_detect_yolov8_rtmp.py \
  --model best.pt \
  --source "rtsp://user:pwd@IP:554/Streaming/Channels/101" \
  --project results --name cam101 \
  --imgsz 1280 --conf 0.30 --device 0 \
  --show --view_scale 0.6 \
  --rtmp_url rtmp://<接收端IP>/live/stream

2) 本地视频文件（不推流，仅显示+保存）
python video_detect_yolov8_rtmp.py --model best.pt --source input.mp4 \
  --project results --name exp1 --show
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2

from ultralytics import YOLO


# ---------------- 工具函数 ----------------
def open_capture(src: str):
    """优先用 FFMPEG 打开，失败则回退默认后端."""
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    return cap


def build_writer(cap, out_path: str):
    """构建本地视频写出器（mp4v），返回 (writer, fps, (w,h))."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 25.0  # RTSP 或部分文件可能取不到 FPS，设个合理默认
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out_path}")
    return writer, fps, (w, h)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def timestamp_values(frame_idx: int, fps: float, cap) -> dict:
    """返回两种时间戳：calc（基于帧率）和 cap（POS_MSEC）."""
    ts_calc = frame_idx / fps if fps > 0 else None
    ts_cap_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    ts_cap = ts_cap_msec / 1000.0 if ts_cap_msec and ts_cap_msec > 0 else None
    return {"calc": ts_calc, "cap": ts_cap}


def start_rtmp_writer(rtmp_url: str, fps: float, w: int, h: int, bitrate: str = "2500k", preset: str = "veryfast"):
    """启动 ffmpeg 子进程，从 stdin 接受 BGR24 原始帧并推送到 RTMP。 - pix_fmt: bgr24 -> yuv420p（编码更通用） - preset: veryfast 负担小；如带宽足可调
    bitrate.
    """
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-re",  # 近实时节奏（避免过载 RTMP 服务器）
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",  # 从 stdin 读
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-tune",
        "zerolatency",
        "-b:v",
        bitrate,
        "-f",
        "flv",
        rtmp_url,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return proc


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="YOLOv8 模型权重 .pt")
    ap.add_argument("--source", required=True, help="视频文件/文件夹/摄像头编号/RTSP/HTTP 地址")
    ap.add_argument("--project", default="runs", help="输出根目录（默认 runs）")
    ap.add_argument("--name", default="predict", help="子目录名称（默认 predict）")
    ap.add_argument("--out_video", default="", help="自定义本地带框视频文件名（可选，如 out.mp4）")
    ap.add_argument("--out_json", default="", help="自定义 JSONL 文件名（可选，如 out.jsonl）")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="如 0 或 cpu，默认自动")
    ap.add_argument("--max_frames", type=int, default=0, help="仅处理前 N 帧（0 表示全部）")
    ap.add_argument("--stride", type=int, default=1, help="处理帧间隔（1=逐帧；2=每2帧一次）")
    ap.add_argument(
        "--timestamp_mode",
        choices=["calc", "cap", "both"],
        default="both",
        help="时间戳来源：calc(帧率)/cap(POS_MSEC)/both（默认）",
    )
    ap.add_argument("--show", action="store_true", help="实时弹窗显示检测画面")
    ap.add_argument("--view_scale", type=float, default=1.0, help="显示窗口缩放（仅显示用）")
    # RTMP 推流相关
    ap.add_argument("--rtmp_url", default="", help="RTMP 推流地址，如 rtmp://<IP>/live/stream")
    ap.add_argument("--rtmp_bitrate", default="2500k", help="视频码率（默认 2500k）")
    ap.add_argument("--rtmp_preset", default="veryfast", help="x264 preset（默认 veryfast）")

    args = ap.parse_args()

    # 输出目录
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 自动命名输出文件（未指定时）
    src_stem = "stream" if str(args.source).startswith(("rtsp://", "http://", "https://")) else Path(args.source).stem
    out_video_path = Path(args.out_video) if args.out_video else out_dir / f"{src_stem}_annot.mp4"
    out_json_path = Path(args.out_json) if args.out_json else out_dir / f"{src_stem}.jsonl"
    ensure_parent(out_video_path)
    ensure_parent(out_json_path)

    # 模型
    model = YOLO(args.model)
    names = model.names  # {id: name}

    # 打开视频/流
    cap = open_capture(args.source)
    writer, fps, (w, h) = build_writer(cap, str(out_video_path))

    # 若需要推 RTMP，启动 ffmpeg 子进程
    rtmp_proc = None
    if args.rtmp_url:
        print(f"[RTMP] Start pushing to: {args.rtmp_url}")
        try:
            rtmp_proc = start_rtmp_writer(args.rtmp_url, fps, w, h, bitrate=args.rtmp_bitrate, preset=args.rtmp_preset)
        except FileNotFoundError:
            raise RuntimeError("未找到 ffmpeg，请安装并加入系统 PATH 后重试。")

    # JSONL
    jf = open(out_json_path, "w", encoding="utf-8")

    frame_idx = 0
    processed = 0
    t0 = time.time()
    t_prev = time.time()
    fps_smoothed = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 帧抽样
            if args.stride > 1 and (frame_idx % args.stride != 0):
                frame_idx += 1
                continue

            # 推理
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            res = results[0]

            # 画框
            annotated = res.plot()

            # 保存本地视频
            writer.write(annotated)

            # RTMP 推流（把 BGR 帧写给 ffmpeg stdin）
            if rtmp_proc and rtmp_proc.stdin:
                try:
                    rtmp_proc.stdin.write(annotated.tobytes())
                except Exception:
                    # 推流中断时不影响主流程；可按需增加重连逻辑
                    pass

            # 组织 JSON（每帧一行）
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy().tolist()
                confs = res.boxes.conf.cpu().numpy().tolist()
                clss = res.boxes.cls.cpu().numpy().tolist()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    c = float(confs[i])
                    cid = int(clss[i])
                    dets.append(
                        {
                            "class_id": cid,
                            "class_name": names.get(cid, str(cid)),
                            "confidence": round(c, 6),
                            "bbox_xyxy": [
                                round(float(x1), 2),
                                round(float(y1), 2),
                                round(float(x2), 2),
                                round(float(y2), 2),
                            ],
                            "bbox_xywh_norm": [
                                round(((x1 + x2) / 2) / w, 6),
                                round(((y1 + y2) / 2) / h, 6),
                                round((x2 - x1) / w, 6),
                                round((y2 - y1) / h, 6),
                            ],
                        }
                    )

            ts = timestamp_values(frame_idx, fps, cap)
            payload = {
                "frame_index": frame_idx,
                "source": args.source,
                "img_size": {"width": w, "height": h},
                "detections": dets,
            }
            if args.timestamp_mode in ("calc", "both"):
                payload["timestamp_calc"] = None if ts["calc"] is None else round(ts["calc"], 3)
            if args.timestamp_mode in ("cap", "both"):
                payload["timestamp_from_cap"] = None if ts["cap"] is None else round(ts["cap"], 3)

            jf.write(json.dumps(payload, ensure_ascii=False) + "\n")

            # 实时显示
            if args.show:
                now = time.time()
                inst = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                alpha = 0.1
                fps_smoothed = inst if fps_smoothed == 0.0 else (alpha * inst + (1 - alpha) * fps_smoothed)

                view = annotated
                if args.view_scale != 1.0:
                    vh, vw = annotated.shape[:2]
                    view = cv2.resize(annotated, (int(vw * args.view_scale), int(vh * args.view_scale)))
                cv2.putText(
                    view,
                    f"FPS: {fps_smoothed:.1f}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("YOLOv8 Live (RTMP)", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            frame_idx += 1
            processed += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                break

    finally:
        cap.release()
        writer.release()
        jf.close()
        if args.show:
            cv2.destroyAllWindows()
        if rtmp_proc:
            try:
                if rtmp_proc.stdin:
                    rtmp_proc.stdin.close()
                rtmp_proc.terminate()
            except Exception:
                pass

    dt = time.time() - t0
    print(f"Done. Frames processed: {processed}, time: {dt:.2f}s, FPS(processed): {processed / max(dt, 1e-6):.2f}")
    print(f"Video saved to: {out_video_path}")
    print(f"JSONL saved to:  {out_json_path}")
    if args.rtmp_url:
        print(f"RTMP pushed to: {args.rtmp_url}")


if __name__ == "__main__":
    main()
