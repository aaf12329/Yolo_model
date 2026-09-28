# -*- coding: utf-8 -*-
"""注视方向数据采集工具 / Gaze-direction data capture tool.

给 EyeWheelchairProject 录制"上看/中看/下看"训练素材：
/ Records "look up / center / down" training footage for the gaze model:
- 屏幕提示当前 zone，倒计时 / on-screen prompt + countdown per zone
- 视频存到 EyeWheelchairProject/data/raw_videos/（不进 git）
- 同时写一个同名 .json，记录每段的 {zone, t_start, t_end}（秒）
  供 auto_label_gaze.py 自动打标 / sidecar json for auto-labeling

用法 / Usage:
    python scripts/collect_gaze_video.py                 # 3 zones x 2 rounds x 20s
    python scripts/collect_gaze_video.py --seconds 15 --rounds 3
    按空格开始录制，按 q 退出 / space to start, q to quit

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: cv2, json, time
"""
import argparse
import json
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "gaze_captures"

ZONES = [  # (key, 中文提示, 英文提示)
    ("up", "往上看 (look UP)", "向上看天花板方向"),
    ("center", "看中间 (look CENTER)", "直视摄像头"),
    ("down", "往下看 (look DOWN)", "向下看膝盖方向"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=20, help="每段时长 / seconds per zone")
    ap.add_argument("--rounds", type=int, default=2, help="轮数 / rounds over all zones")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"摄像头打开失败 / cannot open camera {args.camera}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = OUT_DIR / f"gaze_capture_{stamp}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.FourCC(*"mp4v"), args.fps, (width, height))

    print(f"输出 / output: {video_path}")
    print("按 空格 开始录制 / press SPACE to start recording, q to quit")

    recording = False
    segments = []
    t0 = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.perf_counter()

        if recording:
            elapsed = now - t0
            seg_idx = int(elapsed // args.seconds)
            if seg_idx >= len(ZONES) * args.rounds:
                break  # 采满 / done
            zone_key, zone_zh, _ = ZONES[seg_idx % len(ZONES)]
            left = args.seconds - (elapsed % args.seconds)
            if not segments or segments[-1]["zone"] != zone_key:
                segments.append(
                    {"zone": zone_key, "t_start": round(elapsed, 2), "t_end": None}
                )
            segments[-1]["t_end"] = round(elapsed, 2)
            color = {"up": (80, 200, 120), "center": (200, 180, 60), "down": (60, 120, 230)}[
                zone_key
            ]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), color, -1)
            cv2.putText(frame, zone_zh, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
            cv2.putText(
                frame, f"{left:4.1f}s", (frame.shape[1] - 220, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3,
            )
            writer.write(frame)
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (120, 120, 120), -1)
            cv2.putText(
                frame, "SPACE=start q=quit  (REC standby)",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
            )

        cv2.imshow("gaze capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            recording = False
            break
        if key == ord(" ") and not recording:
            recording = True
            t0 = time.perf_counter()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 清理最后一段未收尾的 t_end / close out trailing segment
    if segments and segments[-1]["t_end"] is None:
        segments[-1]["t_end"] = round(time.perf_counter() - t0, 2)
    json_path = video_path.with_suffix(".json")
    json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"分段信息 / segments: {json_path} ({len(segments)} 段)")
    print("完成 / done. 请保持姿势自然、正常眨眼 / keep natural blinks.")


if __name__ == "__main__":
    main()
