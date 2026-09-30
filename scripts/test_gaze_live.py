# -*- coding: utf-8 -*-
"""眼球状态实时查看器 · 双引擎对比版 / Live eye-state viewer, dual-engine.

同屏对比两套"眼球状态"判定引擎（同一帧、同一批关键点）：
    引擎① YOLO（学习法）  ：模型A 判睁/闭 + 模型B 判注视 5 类（本次训练产物）
    引擎② MediaPipe（几何法）：EAR 判睁/闭 + 虹膜位置判方向（无训练依赖，
                              阈值为固定近似值，不受训练域影响）

为什么双引擎（所有者的实验设计）：我们的 YOLO 在实验室域 95%+，但外机实测崩到
27%（域差，见 docs/gaze_test/domain_gap_analysis.md）。几何法不依赖训练数据——
**如果几何法在你的摄像头上正常而 YOLO 崩，就实锤是训练域的问题**；
如果几何法也崩，则是采集/光照/设备本身的问题。一次实验分辨两种假设。

模型：模型 A 判睁/闭眼（models/eye_yolo26n.pt），
      模型 B 判注视方向（默认 models/gaze5_yolo26s.pt，5 类含左右；--model 切 3 类）。

**没有绿点、没有校准、没有协议**——启动即持续输出双引擎的当前眼球状态：
    屏幕面板：YOLO 与 MediaPipe 两栏对比 + 一致性标记
    终端：各引擎状态稳定变化时打印一行；双引擎结论不一致时提示

按 Q 退出；按 S 存一帧截图（runs/live_snapshots/，存未镜像原始帧）。

诊断开关：--no-gate（闭眼帧也更新注视）/ --dump-crops N（存裁剪图）/ --no-log（关闭会话记录）。
会话记录（默认开）：docs/gaze_test/sessions/session_<时间戳>/ 逐帧 CSV（含双引擎列）+ 摘要。

文件分三段（结构参考 EyeWheelchairProject/src/interaction/gaze_direction_preview.py）：
  1) 判定逻辑（DirectionAnnouncer + 双眼合并 + MediaPipe 几何判定，纯逻辑可单测）
  2) 摄像头与画面（开摄像头、建识别器、中文双栏面板）
  3) 主流程 main()：读帧 → 双引擎判 → 显 → 播

⚠️ 两条铁律（AGENTS.md / README 已知限制）：
  1. 模型输入必须是未镜像帧（镜像会把看左/看右反转）；镜像只用于显示层。
  2. 几何法的方向判定已按"未镜像帧"校准（水平镜像修正系数 1-hx），勿改。

被谁调用 / Called by: 手动 / manual（有摄像头时跑）
内部调用 / Calls: mediapipe.tasks(VIDEO 模式), ultralytics.YOLO ×2, cv2, PIL
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np  # noqa: F401  （draw_text 的 cv2↔PIL 转换需要）
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
EYE_MODEL = ROOT / "models" / "eye_yolo26n.pt"
GAZE_MODEL_DEFAULT = ROOT / "models" / "gaze5_yolo26s.pt"   # 5 类（含左右）
SNAPSHOT_DIR = ROOT / "runs" / "live_snapshots"
CROP_DIR = ROOT / "docs" / "gaze_test" / "crops"
SESSION_DIR = ROOT / "docs" / "gaze_test" / "sessions"

# ---- 与训练数据同款眼周关键点与扩边 ----
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EXPAND_X, EXPAND_Y = 1.6, 2.2

# ---- MediaPipe 虹膜/眼角关键点（几何引擎用）----
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_CORNERS = (33, 133)
RIGHT_CORNERS = (362, 263)

# ---- 类别的中文与动作映射（3 类模型只有前三个键，一样能用）----
GAZE_ZH = {"look_up": "上看", "look_center": "直视", "look_down": "下看",
           "look_left": "看左", "look_right": "看右"}
GAZE_ACTION = {"look_up": "FORWARD", "look_center": "STOP", "look_down": "BACKWARD",
               "look_left": "TURN_LEFT", "look_right": "TURN_RIGHT"}
EYE_ZH = {"open_eye": "睁眼", "closed_eye": "闭眼"}

# ---- MediaPipe 几何引擎的固定阈值（无校准近似；可 --mp-* 调整）----
MP_EAR_CLOSED = 0.20     # EAR 低于此 = 闭眼（常规 6 点 EAR 经验值；个体有差异）
MP_SIDE_THRES = 0.075    # 水平：偏离中线 0.5 超过此值算看左/看右（gaze_direction_preview 同款）
MP_V_UP = -0.113         # 垂直：眼角连线参考系，虹膜高于此线 = 上看（Columbia 真值实测）
MP_V_DOWN = -0.155       # 垂直：虹膜低于此线 = 下看（Columbia 真值实测）

CHANGE_STABLE_FRAMES = 2   # 新方向连续出现这么多帧才播报（防单帧抖动）
CAMERA_INDEX = 0
CAMERA_W = 1280
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)


# ============================ 1) 判定逻辑（纯逻辑） ============================


class DirectionAnnouncer:
    """方向变更播报器：只在"稳定确认的新方向"出现时产出一行，不逐帧刷屏。

    三条规则（每条都能单独测）：
      1. 无效帧（无人脸/闭眼）不参与判断，也不打断已播报的方向；
      2. 新方向必须连续出现 stable_frames 帧才作数（防单帧抖动）；
      3. 与上次播报相同则不重复播报（状态没变就不打印）。
    """

    def __init__(self, stable_frames: int = CHANGE_STABLE_FRAMES) -> None:
        self._stable_frames = stable_frames
        self.current: str | None = None
        self._pending: str | None = None
        self._count = 0

    def update(self, gaze_pred, conf: float, valid: bool) -> str | None:
        if not valid or gaze_pred is None:
            return None                     # 无效帧：不判断，也不清空已播报状态
        if gaze_pred == self.current:
            self._pending, self._count = None, 0
            return None
        if gaze_pred == self._pending:
            self._count += 1
        else:
            self._pending, self._count = gaze_pred, 1
        if self._count < self._stable_frames:
            return None
        previous, self.current = self.current, gaze_pred
        self._pending, self._count = None, 0
        action = GAZE_ACTION.get(gaze_pred, "STOP(fail-safe)")
        return (f"[方向变更] {previous or '--'} → {GAZE_ZH.get(gaze_pred, gaze_pred)}"
                f"({gaze_pred})  conf={conf:.2f}  动作={action}")


def eye_crops(frame, lms):
    """裁左右眼（与 predict.py / 训练数据同款扩边：x1.6 / y2.2）。"""
    h, w = frame.shape[:2]
    crops = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in idx]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half_w = max((max(xs) - min(xs)) / 2 * EXPAND_X, 12)
        half_h = max((max(ys) - min(ys)) / 2 * EXPAND_Y, 12)
        x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
        x1, y1 = min(w, int(cx + half_w)), min(h, int(cy + half_h))
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            crops.append(frame[y0:y1, x0:x1])
    return crops


def top_pred(model, crop, device, conf_thres):
    """单个裁剪 → 最可信类别 + 置信度（与 predict.py 同款）。"""
    r = model.predict(crop, imgsz=128, device=device, verbose=False, conf=conf_thres)[0]
    if not len(r.boxes):
        return None, 0.0
    top = int(r.boxes.conf.argmax())
    return r.names[int(r.boxes.cls[top])], float(r.boxes.conf[top])


def combine(per_eye):
    """双眼取一致；不一致取置信度高者（与 predict.py 同款）。"""
    per_eye = [(c, f) for c, f in per_eye if c is not None]
    if not per_eye:
        return None, 0.0
    if len(per_eye) == 2 and per_eye[0][0] == per_eye[1][0]:
        return per_eye[0][0], min(per_eye[0][1], per_eye[1][1])
    return max(per_eye, key=lambda t: t[1])


# ---- MediaPipe 几何引擎（全部只吃关键点，可单测）----


def ear_ratio(lms) -> float | None:
    """EAR（眼纵横比）：上下眼睑平均距离 ÷ 眼角宽度。闭眼时趋近 0。
    公式与 EyeWheelchairProject/blink_preview.py 同款（6 点：两内两外上下睑）。"""
    vals = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        p = [lms[i] for i in idx]
        horizontal = ((p[3].x - p[0].x) ** 2 + (p[3].y - p[0].y) ** 2) ** 0.5
        if horizontal < 1e-5:
            continue
        d1 = ((p[1].x - p[5].x) ** 2 + (p[1].y - p[5].y) ** 2) ** 0.5
        d2 = ((p[2].x - p[4].x) ** 2 + (p[2].y - p[4].y) ** 2) ** 0.5
        vals.append((d1 + d2) / (2 * horizontal))
    return sum(vals) / len(vals) if vals else None


def gaze_score_x(lms) -> float | None:
    """虹膜在眼角间的水平归一化位置：0=贴左眼角，1=贴右眼角。
    公式与 gaze_direction_preview.py 同款。注意：输入为未镜像帧时，
    使用者往自己的左边看 → 虹膜偏向画面右侧 → 数值偏大（分类时已处理）。"""
    ratios = []
    for iris, corners in ((LEFT_IRIS, LEFT_CORNERS), (RIGHT_IRIS, RIGHT_CORNERS)):
        a, b = lms[corners[0]].x, lms[corners[1]].x
        span = abs(b - a)
        if span < 1e-5:
            continue
        cx = sum(lms[i].x for i in iris) / len(iris)
        ratios.append((cx - min(a, b)) / span)
    return sum(ratios) / len(ratios) if ratios else None


def gaze_vert(lms) -> float | None:
    """虹膜中心相对内外眼角连线的高度（按眼宽归一化，双眼均值）。
    数值越大=虹膜越高=越往上看。Columbia 真值实测三区间零重叠：
    上≈-0.09 > 中≈-0.14 > 下≈-0.17（单受试者标定，阈值可 --mp-* 调整）。"""
    vals = []
    for a_i, b_i, ic in ((LEFT_CORNERS[0], LEFT_CORNERS[1], LEFT_IRIS[0]),
                         (RIGHT_CORNERS[0], RIGHT_CORNERS[1], RIGHT_IRIS[0])):
        a, b, iris = lms[a_i], lms[b_i], lms[ic]
        width = abs(a.x - b.x)
        if width > 1e-6:
            vals.append((iris.y - (a.y + b.y) / 2) / width)
    return sum(vals) / len(vals) if vals else None


def mp_engine(lms, ear_closed: float, side_thres: float, v_up: float, v_down: float):
    """几何引擎：关键点 → (ear, mp_eye, mp_gaze)。mp_eye/mp_gaze 可为 None（不可用）。"""
    ear = ear_ratio(lms)
    mp_eye = None if ear is None else ("closed_eye" if ear < ear_closed else "open_eye")

    hx = gaze_score_x(lms)
    if hx is not None:
        hx = 1.0 - hx   # 未镜像帧修正：使用者看左 → 虹膜在画面右侧 → 镜像坐标后才是"看左"
    vert = gaze_vert(lms)

    mp_gaze = None
    if vert is not None and hx is not None:
        if vert > v_up:
            mp_gaze = "look_up"
        elif vert < v_down:
            mp_gaze = "look_down"
        elif hx < 0.5 - side_thres:
            mp_gaze = "look_left"
        elif hx > 0.5 + side_thres:
            mp_gaze = "look_right"
        else:
            mp_gaze = "look_center"
    return ear, mp_eye, mp_gaze


# ============================ 2) 摄像头与画面 ============================


def open_camera(index: int):
    """打开摄像头：先 DirectShow（Windows 成功率高），失败回退默认后端。"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    if not cap.isOpened():
        raise RuntimeError(f"摄像头 {index} 无法打开：关闭占用它的软件，或换 --camera 1")
    return cap


def create_landmarker():
    """建 MediaPipe 人脸识别器（VIDEO 模式，478 点，含虹膜）。"""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        num_faces=1,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


_font_cache: dict[tuple[Path, int], "ImageFont.FreeTypeFont"] = {}


def chinese_font(size: int):
    """找第一个可用中文字体并缓存（OpenCV 自带字体画不了中文，用 PIL 代替）。"""
    for path in FONT_CANDIDATES:
        if not path.exists():
            continue
        cached = _font_cache.get((path, size))
        if cached is None:
            cached = ImageFont.truetype(str(path), size)
            _font_cache[(path, size)] = cached
        return cached
    raise FileNotFoundError("找不到可用中文字体：" + "、".join(str(p) for p in FONT_CANDIDATES))


def draw_text(frame, text, xy, size, color):
    """在画面上画一行中文；不改传入 frame，返回画好的新图。"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(xy, text, font=chinese_font(size), fill=color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_panel(disp, yolo, mp, fps):
    """左上角双栏面板：YOLO 栏 vs MediaPipe 栏 + 一致性标记。
    yolo/mp = dict(eye=, eye_cf=, gaze=, gaze_cf=, gaze_zh=)。"""
    agree = (yolo["eye"] and mp["eye"] and yolo["eye"] == mp["eye"])
    agree_g = (yolo["gaze"] and mp["gaze"] and yolo["gaze"] == mp["gaze"])
    eye_mark = "✓" if (yolo["eye"] and mp["eye"] and yolo["eye"] == mp["eye"]) else "✗"
    gaze_mark = "✓" if agree_g else "✗"

    cv2.rectangle(disp, (0, 0), (560, 210), (0, 0, 0), -1)
    lines = [
        (f"【YOLO】眼睛:{EYE_ZH.get(yolo['eye'], '--')} {yolo['eye_cf']:.2f}  "
         f"注视:{yolo['gaze_zh']} {yolo['gaze_cf']:.2f}", 20, (60, 220, 60)),
        (f"【MediaPipe】眼睛:{EYE_ZH.get(mp['eye'], '--')}(EAR {mp['ear'] if mp['ear'] is not None else '--'})  "
         f"注视:{GAZE_ZH.get(mp['gaze'], '--')}", 20, (120, 200, 255)),
        (f"一致？ 眼睛:{eye_mark}   注视:{gaze_mark}", 24,
         (60, 220, 120) if (eye_mark == "✓" and gaze_mark == "✓") else (0, 120, 255)),
        (f"动作(YOLO): {GAZE_ACTION.get(yolo['gaze'], 'STOP(fail-safe)')}   FPS: {fps:.1f}", 18, (200, 200, 200)),
    ]
    y = 10
    for text, size, color in lines:
        disp = draw_text(disp, text, (10, y), size, color)
        y += size + 10
    return disp


# ============================ 3) 主流程 ============================


def main() -> None:
    ap = argparse.ArgumentParser(description="眼球状态实时查看器（YOLO × MediaPipe 双引擎对比）")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--model", default=str(GAZE_MODEL_DEFAULT),
                    help="注视模型：默认 5 类（含左右）；3 类主控版 models/gaze_yolo26s.pt")
    ap.add_argument("--device", default=None, help="推理设备 / device：默认自动")
    ap.add_argument("--stable-frames", type=int, default=CHANGE_STABLE_FRAMES,
                    help="新方向连续多少帧才播报变化（防抖）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="处理这么多帧后自动退出（0=一直跑，自检用）")
    ap.add_argument("--no-window", action="store_true",
                    help="不弹窗口只打印（自检用 / headless smoke test）")
    ap.add_argument("--no-gate", action="store_true",
                    help="闭眼帧也更新注视输出（默认闭眼时注视不更新——方向无物理意义）")
    ap.add_argument("--dump-crops", type=int, default=0,
                    help="每 N 帧把眼部裁剪图存 docs/gaze_test/crops/（诊断模型输入质量）")
    ap.add_argument("--no-log", action="store_true",
                    help="不写会话过程记录（默认写：逐帧 CSV（含双引擎列）+ 摘要）")
    ap.add_argument("--mp-ear-closed", type=float, default=MP_EAR_CLOSED)
    ap.add_argument("--mp-side-thres", type=float, default=MP_SIDE_THRES)
    ap.add_argument("--mp-v-up", type=float, default=MP_V_UP)
    ap.add_argument("--mp-v-down", type=float, default=MP_V_DOWN)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    import mediapipe as mp
    device = args.device or (0 if torch.cuda.is_available() else "cpu")

    print("加载模型 / loading models...")
    eye_model = YOLO(str(EYE_MODEL))
    gaze_model = YOLO(str(args.model))
    print(f"注视模型类别 / gaze classes: {gaze_model.names}")
    landmarker = create_landmarker()
    announcer = DirectionAnnouncer(args.stable_frames)
    print(f"设备 / device: {device}")

    cap = open_camera(args.camera)
    print("双引擎查看器已启动（YOLO × MediaPipe 同帧对比）：Q 退出，S 存截图。")
    print(f"终端播报：各引擎状态稳定变化（连续 {args.stable_frames} 帧确认）"
          f"+ 双引擎结论不一致时提示。")

    ts_ms, frame_count, crop_count = 0, 0, 0
    t0 = time.perf_counter()
    yolo = {"eye": None, "eye_cf": 0.0, "gaze": None, "gaze_cf": 0.0, "gaze_zh": "--"}
    mp = {"eye": None, "ear": None, "gaze": None}
    last_disagree = None                    # 上一次"不一致"的描述（用于只打印变化）
    ts_rows: list[list] = []                # 会话逐帧记录
    events: list[tuple[float, str]] = []    # 会话播报事件
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = SESSION_DIR / f"session_{stamp}"
    if not args.no_log:
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"会话记录中 / logging to: {session_dir}")

    try:
        while True:
            # ---- 看：取一帧（模型吃未镜像帧；镜像只用于显示）----
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败 / camera read failed")
            now = time.perf_counter()
            ts_ms += 33
            h, w = frame.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(mp_img, ts_ms)
            lms = res.face_landmarks[0] if res.face_landmarks else None

            # ---- 判：YOLO 双模型 ----
            if lms is not None:
                crops = eye_crops(frame, lms)
                if crops:
                    eye_preds = [top_pred(eye_model, c, device, args.conf) for c in crops]
                    gaze_preds = [top_pred(gaze_model, c, device, args.conf) for c in crops]
                    yolo["eye"], yolo["eye_cf"] = combine(eye_preds)
                    yolo["gaze"], yolo["gaze_cf"] = combine(gaze_preds)
                    yolo["gaze_zh"] = GAZE_ZH.get(yolo["gaze"], yolo["gaze"] or "--")
                if args.dump_crops and frame_count % args.dump_crops == 0:
                    CROP_DIR.mkdir(parents=True, exist_ok=True)
                    for side, c in zip(("L", "R"), eye_crops(frame, lms)):
                        cv2.imwrite(str(CROP_DIR / f"f{frame_count:05d}_{side}.jpg"), c)
                        crop_count += 1
            else:
                yolo.update(eye=None, eye_cf=0.0, gaze=None, gaze_cf=0.0, gaze_zh="--")

            # ---- 判：MediaPipe 几何引擎（同一批关键点，零额外推理成本）----
            if lms is not None:
                mp["ear"], mp["eye"], mp["gaze"] = mp_engine(
                    lms, args.mp_ear_closed, args.mp_side_thres,
                    args.mp_v_up, args.mp_v_down)
            else:
                mp.update(eye=None, ear=None, gaze=None)

            frame_count += 1
            fps = frame_count / max(now - t0, 0.001)

            # ---- 显：镜像显示（体感）+ 双栏面板 ----
            if not args.no_window:
                disp = cv2.flip(frame, 1)
                disp = draw_panel(disp, yolo, mp, fps)
                cv2.imshow("dual engine | Q quit | S snapshot", disp)

            # ---- 播：YOLO 状态变化 + 双引擎不一致提示（均只在变化时打印）----
            valid = (yolo["eye"] == "open_eye") or args.no_gate
            line = announcer.update(yolo["gaze"], yolo["gaze_cf"],
                                    valid=valid and yolo["gaze"] is not None)
            if line:
                events.append((round(now - t0, 1), line))
                print(f"{line}   @ {now - t0:.1f}s")
            both = yolo["eye"] is not None and mp["eye"] is not None
            dis = None
            if both:
                if yolo["eye"] != mp["eye"]:
                    dis = f"眼睛不一致: YOLO={EYE_ZH[yolo['eye']]} vs MP=({'睁眼' if mp['eye']=='open_eye' else '闭眼'})"
                elif yolo["gaze"] and mp["gaze"] and yolo["gaze"] != mp["gaze"]:
                    dis = (f"注视不一致: YOLO={GAZE_ZH.get(yolo['gaze'], yolo['gaze'])} vs "
                           f"MP={GAZE_ZH.get(mp['gaze'], mp['gaze'])}")
            if dis != last_disagree:
                if dis:
                    print(f"[双引擎分歧] {dis}   @ {now - t0:.1f}s")
                last_disagree = dis

            # ---- 会话记录 ----
            if not args.no_log:
                ts_rows.append([round(now - t0, 2), frame_count,
                                yolo["eye"] or "no_face", round(yolo["eye_cf"], 3),
                                yolo["gaze"] or "no_face", round(yolo["gaze_cf"], 3),
                                None if mp["ear"] is None else round(mp["ear"], 4),
                                mp["eye"] or "no_face", mp["gaze"] or "no_face"])

            if args.max_frames and frame_count >= args.max_frames:
                print(f"已处理 {frame_count} 帧，自动退出（--max-frames）")
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                snap = SNAPSHOT_DIR / f"live_{int(time.time())}.jpg"
                cv2.imwrite(str(snap), frame)   # 存未镜像原始帧
                print(f"截图已存 / snapshot: {snap}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if crop_count:
            print(f"诊断裁剪已存 / diag crops: {CROP_DIR}（{crop_count} 张）")
        # ---- 会话收尾：逐帧 CSV（含双引擎列）+ 摘要 ----
        if not args.no_log and ts_rows:
            elapsed = time.perf_counter() - t0
            session_dir.mkdir(parents=True, exist_ok=True)
            csv_path = session_dir / f"session_{stamp}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "frame", "yolo_eye", "yolo_eye_conf",
                            "yolo_gaze", "yolo_gaze_conf",
                            "mp_ear", "mp_eye", "mp_gaze"])
                w.writerows(ts_rows)

            n = len(ts_rows)
            eye_agree = sum(1 for r in ts_rows
                            if r[2] != "no_face" and r[7] != "no_face" and r[2] == r[7])
            gaze_agree = sum(1 for r in ts_rows
                             if r[4] != "no_face" and r[8] != "no_face" and r[4] == r[8])
            comparable_eye = sum(1 for r in ts_rows if r[2] != "no_face" and r[7] != "no_face")
            comparable_gaze = sum(1 for r in ts_rows if r[4] != "no_face" and r[8] != "no_face")
            yolo_gaze_counter = Counter(r[4] for r in ts_rows if r[4] != "no_face")

            md = [
                f"# 双引擎会话记录 / dual-engine session {stamp}",
                "",
                f"- 时长 / duration: {elapsed:.1f}s　帧数 / frames: {n}"
                f"（平均 {n / max(elapsed, 0.001):.1f} FPS）",
                f"- 模型 / models: eye={EYE_MODEL.name}, gaze={Path(args.model).name}",
                f"- YOLO 注视分布 / yolo gaze: " + "，".join(
                    f"{GAZE_ZH.get(k, k)}={v}" for k, v in yolo_gaze_counter.most_common()),
                f"- 双引擎一致率（可比较帧）: 眼睛 "
                f"{eye_agree}/{comparable_eye} = {eye_agree / max(1, comparable_eye) * 100:.1f}% ｜ "
                f"注视 {gaze_agree}/{comparable_gaze} = {gaze_agree / max(1, comparable_gaze) * 100:.1f}%",
                f"- 诊断裁剪 / crops: {crop_count} 张" + (f"（--dump-crops {args.dump_crops}）" if args.dump_crops else ""),
                "",
                "## 状态变化时间线 / state-change timeline（YOLO + 分歧提示）",
                "",
            ]
            md += [f"- {t:.1f}s　{line}" for t, line in events] or ["-（无状态变化 / no changes）"]
            md += ["", f"- 逐帧数据 / per-frame: `{csv_path.name}`（含 mp_ear/mp_eye/mp_gaze 列）",
                   f"- 原始截图（如按过 S）: `runs/live_snapshots/`"]
            summary_path = session_dir / f"session_{stamp}_summary.md"
            summary_path.write_text("\n".join(md), encoding="utf-8")
            print(f"\n会话记录已保存 / session log saved:")
            print(f"  逐帧 CSV: {csv_path}")
            print(f"  摘要:     {summary_path}")


if __name__ == "__main__":
    main()
