import streamlit as st
import cv2
import numpy as np
import io
import os
import re
import csv
import json
import math
import random
import time
import base64
import tempfile
import hashlib
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
    PIL_AVAILABLE = True
    try:
        PIL_LANCZOS = Image.Resampling.LANCZOS
    except AttributeError:
        PIL_LANCZOS = Image.LANCZOS
except ImportError:
    PIL_AVAILABLE = False
    PIL_LANCZOS = None

st.set_page_config(
    page_title="Autonomous Multi-Object Tracking Engine",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
WIDTH = 960
HEIGHT = 540
FPS = 15
MARGIN = 170

GIF_WIDTH = 480
GIF_HEIGHT = 270

COLOR_MAP = {
    "blue": (255, 130, 30),
    "red": (45, 45, 220),
    "green": (55, 185, 55),
    "yellow": (35, 200, 230),
    "brown": (55, 95, 145),
    "purple": (180, 65, 160),
    "white": (235, 235, 235),
    "gray": (130, 130, 130),
    "black": (35, 35, 35),
    "orange": (35, 145, 240),
}

PALETTE = [
    (255, 128, 0),
    (128, 255, 0),
    (0, 255, 128),
    (0, 128, 255),
    (128, 0, 255),
    (255, 0, 128),
    (0, 255, 255),
    (255, 255, 0),
    (180, 105, 255),
    (255, 180, 0),
    (0, 180, 255),
    (180, 255, 0),
]

COLOR_WORDS = [
    "red", "blue", "green", "yellow", "brown",
    "purple", "white", "gray", "black", "orange"
]

CLASS_SYNONYMS = {
    "person": ["person", "people", "human", "player"],
    "vehicle": ["vehicle", "car", "truck", "forklift"],
    "forklift": ["forklift"],
    "animal": ["animal", "dog", "wildlife", "creature"],
    "ball": ["ball"],
    "backpack": ["backpack", "bag"],
    "container": ["container", "door", "crate"],
    "shelf_item": ["item", "product", "shelf", "stock"],
}

SCENARIOS = [
    "integrated",
    "occlusion",
    "reid",
    "camera",
    "openvocab",
    "state",
    "sports",
    "wildlife",
    "retail",
    "multicamera",
]

DEFAULT_PARAMS = {
    "scenario": "integrated",
    "algorithm": "BoT-SORT",
    "confidence": 0.45,
    "iou": 0.25,
    "max_age": 12,
    "appearance_weight": 0.30,
    "reid_threshold": 0.52,
    "camera_comp": True,
    "det_noise": 0.25,
    "duration": 10,
    "query": "red backpack",
    "benchmark": True,
    "seed": 7,
}


# ------------------------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------------------------
def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> List[int]:
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    return [x1, y1, x2, y2]


def xyxy_to_cwh(box: List[int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1))


def shift_box(box: List[int], dx: int, dy: int) -> List[int]:
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def center_of_box(box: List[int]) -> Tuple[int, int]:
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))


def intersection_area(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def box_iou(a: List[int], b: List[int]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    aw = max(1, a[2] - a[0])
    ah = max(1, a[3] - a[1])
    bw = max(1, b[2] - b[0])
    bh = max(1, b[3] - b[1])
    union = float(aw * ah + bw * bh - inter)
    return inter / union if union > 0 else 0.0


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 1.0
    a = l2_normalize(a)
    b = l2_normalize(b)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    cos = float(np.dot(a, b) / (na * nb))
    cos = clamp(cos, -1.0, 1.0)
    return 1.0 - cos


def id_color(track_id: int) -> Tuple[int, int, int]:
    return PALETTE[int(track_id) % len(PALETTE)]


def parse_query(q: str) -> Tuple[List[str], List[str]]:
    q = (q or "").lower()
    tokens = re.findall(r"[a-z]+", q)
    token_set = set(tokens)

    colors = [c for c in COLOR_WORDS if c in token_set]
    classes = []
    for canon, syns in CLASS_SYNONYMS.items():
        if any(s in token_set for s in syns):
            classes.append(canon)
    return colors, classes


def class_matches(query_cls: str, object_cls: str) -> bool:
    if query_cls == object_cls:
        return True
    if query_cls == "vehicle" and object_cls in ("vehicle", "forklift"):
        return True
    if query_cls == "forklift" and object_cls == "forklift":
        return True
    if query_cls == "container" and object_cls in ("container", "shelf_item"):
        return True
    if query_cls == "shelf_item" and object_cls in ("shelf_item", "container"):
        return True
    return False


def query_match_score(object_cls: str, object_color: str,
                      colors: List[str], classes: List[str]) -> float:
    if not colors and not classes:
        return 0.0
    score = 0.0
    total = 0
    if colors:
        total += 1
        if object_color in colors:
            score += 1.0
    if classes:
        total += 1
        if any(class_matches(c, object_cls) for c in classes):
            score += 1.0
    return score / max(1, total)


def embedding_from_crop(frame: np.ndarray, box: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1 = int(clamp(x1, 0, w - 1))
    y1 = int(clamp(y1, 0, h - 1))
    x2 = int(clamp(x2, 1, w))
    y2 = int(clamp(y2, 1, h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros(32, dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    feats: List[float] = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [8], [0, 256]).flatten()
        s = float(hist.sum())
        if s > 0:
            hist = hist / s
        feats.extend(hist.tolist())

    flat = hsv.reshape(-1, 3).astype(np.float32)
    mean = flat.mean(axis=0) / 255.0
    std = flat.std(axis=0) / 255.0
    feats.extend(mean.tolist())
    feats.extend(std.tolist())

    bw = x2 - x1
    bh = y2 - y1
    feats.append(float(bw) / max(1.0, float(bh)))
    feats.append(float(math.log1p(bw * bh)) / 10.0)

    return l2_normalize(np.array(feats, dtype=np.float32))


def draw_transparent_rect(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], alpha: float = 0.5) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_dashed_rectangle(img: np.ndarray, box: List[int],
                          color: Tuple[int, int, int], thickness: int = 2,
                          dash: int = 8) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)


def _as_points(pts: Optional[np.ndarray]) -> np.ndarray:
    """
    Normalize OpenCV point arrays to shape (N, 2).

    OpenCV functions such as goodFeaturesToTrack and calcOpticalFlowPyrLK may
    return arrays shaped like:
      - (N, 2)
      - (N, 1, 2)
      - (1, N, 2)
      - empty arrays
      - None

    This helper prevents IndexError such as:
      index 1 is out of bounds for axis 1 with size 1
    """
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)

    arr = np.asarray(pts, dtype=np.float32)

    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == 2:
            return arr.reshape(1, 2)
        if arr.size % 2 == 0:
            return arr.reshape(-1, 2)
        return np.empty((0, 2), dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            return arr[:, :2].copy()
        if arr.shape[1] == 1:
            flat = arr.reshape(-1)
            if flat.size % 2 == 0:
                return flat.reshape(-1, 2)
        return np.empty((0, 2), dtype=np.float32)

    if arr.ndim == 3:
        # Common OpenCV shape: (N, 1, 2)
        if arr.shape[2] >= 2:
            return arr.reshape(-1, arr.shape[2])[:, :2].copy()
        # Alternative shape: (1, N, 2)
        if arr.shape[1] >= 2:
            return arr.reshape(-1, arr.shape[1])[:, :2].copy()
        flat = arr.reshape(-1)
        if flat.size % 2 == 0:
            return flat.reshape(-1, 2)

    return np.empty((0, 2), dtype=np.float32)


# ------------------------------------------------------------------------------
# Synthetic scene generation
# ------------------------------------------------------------------------------
def make_background(W: int, H: int, scenario: str) -> np.ndarray:
    bigW = W + 2 * MARGIN
    bigH = H + 2 * MARGIN
    img = np.zeros((bigH, bigW, 3), dtype=np.uint8)

    # Gradient base
    top = np.array([165, 150, 135], dtype=np.float32)
    bottom = np.array([75, 88, 100], dtype=np.float32)
    for y in range(bigH):
        t = y / max(1, bigH - 1)
        c = (1 - t) * top + t * bottom
        img[y, :] = np.clip(c, 0, 255).astype(np.uint8)

    # Static texture points for optical flow / camera-motion estimation
    rng = np.random.default_rng(42)
    for _ in range(3200):
        x = int(rng.integers(0, bigW))
        y = int(rng.integers(0, bigH))
        r = int(rng.integers(1, 3))
        b = int(rng.integers(95, 175))
        g = int(rng.integers(95, 175))
        rr = int(rng.integers(95, 175))
        cv2.circle(img, (x, y), r, (b, g, rr), -1, cv2.LINE_AA)

    # Subtle grid
    for x in range(0, bigW, 90):
        cv2.line(img, (x, 0), (x, bigH), (110, 110, 110), 1, cv2.LINE_AA)
    for y in range(0, bigH, 90):
        cv2.line(img, (0, y), (bigW, y), (110, 110, 110), 1, cv2.LINE_AA)

    if scenario in ("sports", "integrated"):
        cv2.rectangle(img, (MARGIN + 40, MARGIN + 260), (MARGIN + W - 40, MARGIN + H - 30), (210, 210, 210), 2)
        cv2.circle(img, (MARGIN + W // 2, MARGIN + 390), 70, (210, 210, 210), 2)
        cv2.line(img, (MARGIN + W // 2, MARGIN + 260), (MARGIN + W // 2, MARGIN + H - 30), (210, 210, 210), 2)

    if scenario in ("retail", "integrated"):
        for i in range(3):
            x1 = MARGIN + 690
            y1 = MARGIN + 55 + i * 55
            x2 = MARGIN + 910
            y2 = y1 + 38
            cv2.rectangle(img, (x1, y1), (x2, y2), (95, 95, 95), 2)
            cv2.putText(img, f"SHELF {i+1}", (x1 + 5, y1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 185, 185), 1, cv2.LINE_AA)

    if scenario in ("wildlife", "integrated"):
        for _ in range(90):
            x = int(rng.integers(0, bigW))
            y = int(rng.integers(bigH // 2, bigH))
            r = int(rng.integers(12, 38))
            cv2.circle(img, (x, y), r, (55, 95, 55), -1, cv2.LINE_AA)

    if scenario in ("multicamera", "integrated"):
        cx = MARGIN + W // 2
        for y in range(0, bigH, 18):
            cv2.line(img, (cx, y), (cx, min(y + 10, bigH)), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "CAM A", (MARGIN + 20, MARGIN + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "CAM B", (MARGIN + W - 100, MARGIN + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    return img


def make_objects(scenario: str, total_frames: int) -> List[Dict[str, Any]]:
    last = max(1, total_frames - 1)
    objs: List[Dict[str, Any]] = []

    def add(oid: int, cls: str, color: str, w: int, h: int,
            segments: List[Tuple[Any, ...]],
            state_events: Optional[List[Tuple[int, str]]] = None,
            zone: Optional[str] = None,
            species: Optional[str] = None,
            default_state: Optional[str] = None) -> None:
        segs = []
        for s in segments:
            if len(s) == 6:
                s0, e0, x0, y0, x1, y1 = s
                opts = {}
            elif len(s) == 7:
                s0, e0, x0, y0, x1, y1, opts = s
            else:
                raise ValueError("Bad segment")
            segs.append({
                "start": int(s0 * last),
                "end": int(e0 * last),
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "bounce_amp": float(opts.get("bounce_amp", 0.0)),
                "bounce_freq": float(opts.get("bounce_freq", 0.0)),
                "sway_amp": float(opts.get("sway_amp", 0.0)),
                "sway_freq": float(opts.get("sway_freq", 0.0)),
            })

        if default_state is None:
            default_state = state_events[0][1] if state_events else "normal"

        objs.append({
            "id": oid,
            "cls": cls,
            "color": color,
            "color_bgr": COLOR_MAP.get(color, (120, 120, 120)),
            "w": int(w),
            "h": int(h),
            "segments": segs,
            "state_events": state_events or [],
            "default_state": default_state,
            "zone": zone,
            "species": species,
        })

    # Full integrated catalogue
    add(1, "person", "blue", 55, 130,
        [(0.00, 0.30, 120, 300, 780, 300),
         (0.58, 1.00, 100, 310, 820, 310)])

    add(2, "person", "red", 52, 125,
        [(0.15, 1.00, 180, 260, 720, 250, {"sway_amp": 8.0, "sway_freq": 0.18})])

    add(3, "person", "green", 54, 128,
        [(0.15, 1.00, 760, 270, 220, 260, {"sway_amp": 8.0, "sway_freq": 0.17})])

    add(4, "person", "purple", 50, 120,
        [(0.00, 1.00, 850, 330, 120, 340)])

    add(5, "forklift", "yellow", 150, 70,
        [(0.00, 1.00, 80, 430, 880, 430)])

    add(6, "animal", "brown", 65, 40,
        [(0.00, 1.00, 200, 480, 820, 470, {"bounce_amp": 16.0, "bounce_freq": 0.35})],
        species="dog")

    add(7, "ball", "white", 26, 26,
        [(0.65, 1.00, 100, 390, 860, 360, {"bounce_amp": 45.0, "bounce_freq": 0.55})])

    add(8, "container", "gray", 95, 110,
        [(0.00, 1.00, 835, 150, 835, 150)],
        state_events=[(0, "closed"), (int(0.70 * last), "open"), (int(0.88 * last), "closed")],
        default_state="closed")

    add(9, "backpack", "red", 35, 45,
        [(0.32, 1.00, 300, 470, 700, 470, {"sway_amp": 4.0, "sway_freq": 0.22})])

    add(10, "shelf_item", "blue", 40, 30,
        [(0.00, 0.76, 760, 90, 760, 90),
         (0.90, 1.00, 760, 90, 760, 90)],
        zone="shelf_A")

    if scenario == "wildlife":
        add(11, "animal", "black", 55, 35,
            [(0.00, 1.00, 820, 500, 180, 490, {"bounce_amp": 10.0, "bounce_freq": 0.28})],
            species="fox")
        add(12, "animal", "gray", 45, 30,
            [(0.20, 1.00, 120, 510, 780, 500, {"bounce_amp": 8.0, "bounce_freq": 0.31})],
            species="rabbit")

    scenario_ids = {
        "integrated": set(range(1, 11)),
        "occlusion": {1, 2, 3, 4},
        "reid": {1, 4, 9},
        "camera": {1, 2, 5, 6},
        "openvocab": {2, 3, 5, 9},
        "state": {5, 8, 10},
        "sports": {2, 3, 4, 7},
        "wildlife": {6, 11, 12},
        "retail": {5, 8, 10},
        "multicamera": {1, 4, 9},
    }

    if scenario in scenario_ids:
        keep = scenario_ids[scenario]
        objs = [o for o in objs if o["id"] in keep]

    if not objs:
        objs = [o for o in objs if o["id"] in scenario_ids["integrated"]]

    return objs


def object_pose(obj: Dict[str, Any], f: int) -> Optional[Dict[str, Any]]:
    for s in obj["segments"]:
        if s["start"] <= f <= s["end"]:
            denom = max(1, s["end"] - s["start"])
            p = (f - s["start"]) / float(denom)
            cx = s["x0"] + (s["x1"] - s["x0"]) * p
            cy = s["y0"] + (s["y1"] - s["y0"]) * p

            if s.get("sway_amp", 0.0):
                cx += s["sway_amp"] * math.sin(s.get("sway_freq", 0.0) * f)
            if s.get("bounce_amp", 0.0):
                cy -= s["bounce_amp"] * abs(math.sin(s.get("bounce_freq", 0.0) * f))

            scale = 1.0 + 0.00025 * (cy - 300.0)
            scale = clamp(scale, 0.72, 1.35)
            w = max(8, int(obj["w"] * scale))
            h = max(8, int(obj["h"] * scale))

            state = obj.get("default_state", "normal")
            for ef, est in obj.get("state_events", []):
                if f >= ef:
                    state = est

            return {"cx": float(cx), "cy": float(cy), "w": int(w), "h": int(h), "state": state}
    return None


def absolute_camera_offset(f: int, scenario: str, total_frames: int) -> Tuple[int, int]:
    dx = 2.0 * math.sin(2 * math.pi * f / 77.0)
    dy = 1.2 * math.cos(2 * math.pi * f / 61.0)

    if scenario == "camera":
        dx += 55.0 * math.sin(2 * math.pi * f / 95.0) + 12.0 * math.sin(2 * math.pi * f / 37.0)
        dy += 25.0 * math.cos(2 * math.pi * f / 83.0) + 6.0 * math.sin(2 * math.pi * f / 29.0)
    elif scenario == "integrated" and f > total_frames * 0.50:
        ramp = min(1.0, (f - total_frames * 0.50) / (total_frames * 0.12 + 1.0))
        dx += ramp * (45.0 * math.sin(2 * math.pi * f / 90.0) + 10.0 * math.sin(2 * math.pi * f / 35.0))
        dy += ramp * (20.0 * math.cos(2 * math.pi * f / 75.0) + 5.0 * math.sin(2 * math.pi * f / 27.0))

    return int(round(dx)), int(round(dy))


def draw_object(frame: np.ndarray, obj: Dict[str, Any], cx: float, cy: float,
                w: int, h: int, state: str, f: int) -> None:
    x1, y1, x2, y2 = xywh_to_xyxy(cx, cy, w, h)
    color = obj["color_bgr"]
    cls = obj["cls"]

    # Soft shadow
    cv2.ellipse(frame, (int(cx), int(y2 - 2)), (int(w * 0.42), 5), 0, 0, 360, (40, 40, 40), -1, cv2.LINE_AA)

    if cls == "person":
        body_c = (int(cx), int(cy + h * 0.08))
        cv2.ellipse(frame, body_c, (int(w * 0.34), int(h * 0.43)), 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(cx), int(cy - h * 0.31)), int(w * 0.21), (215, 195, 175), -1, cv2.LINE_AA)
        cv2.line(frame, (int(cx - w * 0.12), int(y2 - h * 0.18)), (int(cx - w * 0.18), y2), (45, 45, 45), 3, cv2.LINE_AA)
        cv2.line(frame, (int(cx + w * 0.12), int(y2 - h * 0.18)), (int(cx + w * 0.18), y2), (45, 45, 45), 3, cv2.LINE_AA)

    elif cls in ("vehicle", "forklift"):
        cv2.rectangle(frame, (x1, y1 + int(h * 0.22)), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1 + int(w * 0.56), y1 + int(h * 0.02)), (x2 - int(w * 0.04), y1 + int(h * 0.48)), (65, 65, 65), -1, cv2.LINE_AA)
        cv2.circle(frame, (x1 + int(w * 0.20), y2 - 4), 9, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - int(w * 0.20), y2 - 4), 9, (25, 25, 25), -1, cv2.LINE_AA)
        if cls == "forklift":
            cv2.rectangle(frame, (x2 - 4, y1 + int(h * 0.35)), (x2 + 22, y1 + int(h * 0.45)), (90, 90, 90), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (x2 - 4, y1 + int(h * 0.62)), (x2 + 22, y1 + int(h * 0.72)), (90, 90, 90), -1, cv2.LINE_AA)

    elif cls == "animal":
        cv2.ellipse(frame, (int(cx), int(cy)), (int(w * 0.44), int(h * 0.34)), 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(cx + w * 0.32), int(cy - h * 0.10)), int(h * 0.22), color, -1, cv2.LINE_AA)
        cv2.line(frame, (int(cx - w * 0.38), int(cy - h * 0.05)), (int(cx - w * 0.58), int(cy - h * 0.25)), color, 3, cv2.LINE_AA)

    elif cls == "ball":
        cv2.circle(frame, (int(cx), int(cy)), int(w / 2), color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(cx - w * 0.12), int(cy - h * 0.12)), int(w * 0.12), (255, 255, 255), -1, cv2.LINE_AA)

    elif cls == "backpack":
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1 + 4, y1 + 6), (x2 - 4, y2 - 8), (35, 35, 35), 2, cv2.LINE_AA)
        cv2.line(frame, (x1 + 6, y1), (x1 + 6, y1 - 8), (35, 35, 35), 2, cv2.LINE_AA)
        cv2.line(frame, (x2 - 6, y1), (x2 - 6, y1 - 8), (35, 35, 35), 2, cv2.LINE_AA)

    elif cls == "container":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (105, 105, 105), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 2, cv2.LINE_AA)
        if state == "open":
            cv2.rectangle(frame, (x1 + 10, y1 + 18), (x2 - 10, y2 - 12), (25, 25, 25), -1, cv2.LINE_AA)
            cv2.line(frame, (x1 + 8, y1 + 16), (x2 - 8, y1 + 2), (180, 180, 180), 4, cv2.LINE_AA)
        else:
            cv2.line(frame, (int((x1 + x2) / 2), y1 + 8), (int((x1 + x2) / 2), y2 - 8), (70, 70, 70), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x2 - 20, int((y1 + y2) / 2) - 8), (x2 - 8, int((y1 + y2) / 2) + 8), (55, 55, 55), -1, cv2.LINE_AA)

    elif cls == "shelf_item":
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (245, 245, 245), 1, cv2.LINE_AA)

    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)


def render_raw(objects: List[Dict[str, Any]], f: int, offset: Tuple[int, int],
               bg_big: np.ndarray, scenario: str) -> np.ndarray:
    ox, oy = offset
    crop_x = int(clamp(MARGIN - ox, 0, bg_big.shape[1] - WIDTH))
    crop_y = int(clamp(MARGIN - oy, 0, bg_big.shape[0] - HEIGHT))
    frame = bg_big[crop_y:crop_y + HEIGHT, crop_x:crop_x + WIDTH].copy()

    draw_list = []
    for obj in objects:
        pose = object_pose(obj, f)
        if pose is not None:
            draw_list.append((pose["cy"], obj, pose))
    draw_list.sort(key=lambda x: x[0])

    for _, obj, pose in draw_list:
        cx = pose["cx"] + ox
        cy = pose["cy"] + oy
        draw_object(frame, obj, cx, cy, pose["w"], pose["h"], pose["state"], f)

    return frame


def simulate_detections(frame: np.ndarray, objects: List[Dict[str, Any]], f: int,
                        offset: Tuple[int, int], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    visible = []
    for obj in objects:
        pose = object_pose(obj, f)
        if pose is not None:
            visible.append((obj, pose))

    dets = []
    rng = random.Random(int(params["seed"]) * 100003 + f)
    noise = float(params["det_noise"])

    for obj, pose in visible:
        cx = pose["cx"] + offset[0]
        cy = pose["cy"] + offset[1]
        w = pose["w"]
        h = pose["h"]
        box = xywh_to_xyxy(cx, cy, w, h)
        area = max(1.0, float(w * h))

        occ = 0.0
        for obj2, pose2 in visible:
            if obj2["id"] == obj["id"]:
                continue
            box2 = xywh_to_xyxy(pose2["cx"] + offset[0], pose2["cy"] + offset[1], pose2["w"], pose2["h"])
            inter = intersection_area(box, box2)
            occ = max(occ, inter / area)

        small = 1.0 if area < 1200 else 0.0
        miss_p = 0.02 + 0.12 * noise + 0.35 * occ + 0.08 * small
        if rng.random() < miss_p:
            continue

        sx = 1.2 + 4.5 * noise
        jx = rng.gauss(0, sx)
        jy = rng.gauss(0, sx)
        jw = rng.randint(-int(2 + 4 * noise), int(2 + 4 * noise))
        jh = rng.randint(-int(2 + 4 * noise), int(2 + 4 * noise))

        x1 = int(clamp(box[0] + jx - jw / 2.0, 0, WIDTH - 2))
        y1 = int(clamp(box[1] + jy - jh / 2.0, 0, HEIGHT - 2))
        x2 = int(clamp(box[2] + jx + jw / 2.0, x1 + 4, WIDTH))
        y2 = int(clamp(box[3] + jy + jh / 2.0, y1 + 4, HEIGHT))

        score = 0.96 - 0.25 * noise - 0.45 * occ - 0.12 * small + rng.uniform(-0.06, 0.04)
        score = float(clamp(score, 0.08, 0.99))

        emb = embedding_from_crop(frame, [x1, y1, x2, y2])

        dets.append({
            "bbox": [x1, y1, x2, y2],
            "score": score,
            "cls": obj["cls"],
            "color": obj["color"],
            "state": pose["state"],
            "embedding": emb,
            "gt_id": obj["id"],
            "species": obj.get("species"),
        })

    return dets


# ------------------------------------------------------------------------------
# Camera motion estimator
# ------------------------------------------------------------------------------
class CameraMotionEstimator:
    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.cum: np.ndarray = np.zeros(2, dtype=np.float32)

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=350, qualityLevel=0.01, minDistance=13, blockSize=5
        )
        return _as_points(pts)

    def estimate(self, gray: np.ndarray) -> Tuple[int, int]:
        gray = np.asarray(gray)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_pts = self._detect(gray)
            return (0, 0)

        if len(self.prev_pts) == 0:
            self.prev_pts = self._detect(gray)
            self.prev_gray = gray.copy()
            return (int(round(float(self.cum[0]))), int(round(float(self.cum[1]))))

        try:
            prev_for_lk = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_for_lk, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
        except cv2.error:
            self.prev_pts = self._detect(gray)
            self.prev_gray = gray.copy()
            return (int(round(float(self.cum[0]))), int(round(float(self.cum[1]))))

        next_pts = _as_points(next_pts)

        if status is None:
            self.prev_pts = self._detect(gray)
            self.prev_gray = gray.copy()
            return (int(round(float(self.cum[0]))), int(round(float(self.cum[1]))))

        status = np.asarray(status).reshape(-1)

        if len(status) != len(self.prev_pts) or len(next_pts) != len(self.prev_pts):
            self.prev_pts = self._detect(gray)
            self.prev_gray = gray.copy()
            return (int(round(float(self.cum[0]))), int(round(float(self.cum[1]))))

        good = status == 1
        gp = self.prev_pts[good]
        gn = next_pts[good]

        if len(gp) >= 20:
            dx = gn[:, 0] - gp[:, 0]
            dy = gn[:, 1] - gp[:, 1]
            mdx = float(np.median(dx))
            mdy = float(np.median(dy))
            madx = float(np.median(np.abs(dx - mdx))) + 1e-6
            mady = float(np.median(np.abs(dy - mdy))) + 1e-6
            mask = (np.abs(dx - mdx) < 5.0 * madx) & (np.abs(dy - mdy) < 5.0 * mady)
            if int(mask.sum()) >= 10:
                mdx = float(np.median(dx[mask]))
                mdy = float(np.median(dy[mask]))
            self.cum += np.array([mdx, mdy], dtype=np.float32)

        if len(gn) < 180:
            new_pts = self._detect(gray)
            if len(new_pts) > 0:
                gn = new_pts if len(gn) == 0 else np.vstack([gn, new_pts])

        if len(gn) == 0:
            gn = self._detect(gray)

        self.prev_pts = np.asarray(gn, dtype=np.float32).reshape(-1, 2)
        if len(self.prev_pts) > 350:
            self.prev_pts = self.prev_pts[:350]
        self.prev_gray = gray.copy()

        return (int(round(float(self.cum[0]))), int(round(float(self.cum[1]))))


# ------------------------------------------------------------------------------
# Tracker
# ------------------------------------------------------------------------------
class Track:
    def __init__(self, track_id: int, det: Dict[str, Any], f: int, offset_used: Tuple[int, int]) -> None:
        self.id = track_id
        self.tlbr_model = [int(v) for v in det["bbox_model"]]
        self.x, self.y, self.w, self.h = xyxy_to_cwh(self.tlbr_model)
        self.vx = 0.0
        self.vy = 0.0
        self.vw = 0.0
        self.vh = 0.0
        self.render_tlbr = shift_box(self.tlbr_model, offset_used[0], offset_used[1])
        self.score = float(det["score"])
        self.cls = det["cls"]
        self.color = det["color"]
        self.embedding = np.array(det["embedding"], dtype=np.float32)
        self.state = det.get("state", "normal")
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.confirmed = False
        self.occluded = False
        self.occluded_start: Optional[int] = None
        self.occlusion_event_sent = False
        self.recovered_after = 0
        self.lost = False
        self.deleted = False
        self.last_frame = f
        self.created_frame = f
        self.reid_conf: Optional[float] = None

    def predict(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.w = clamp(self.w + self.vw, 8.0, 700.0)
        self.h = clamp(self.h + self.vh, 8.0, 700.0)

        self.vx *= 0.92
        self.vy *= 0.92
        self.vw *= 0.88
        self.vh *= 0.88

        self.tlbr_model = xywh_to_xyxy(self.x, self.y, self.w, self.h)
        self.age += 1
        self.time_since_update += 1

        if self.time_since_update == 1:
            self.occluded = True
            self.occluded_start = self.last_frame
        elif self.time_since_update > 1:
            self.occluded = True

    def update(self, det: Dict[str, Any], f: int, offset_used: Tuple[int, int]) -> None:
        was_occluded = self.occluded
        duration = 0
        if was_occluded and self.occluded_start is not None:
            duration = f - self.occluded_start

        nx, ny, nw, nh = xyxy_to_cwh(det["bbox_model"])
        alpha = 0.58 if det["score"] >= 0.60 else 0.36

        dx = nx - self.x
        dy = ny - self.y
        dw = nw - self.w
        dh = nh - self.h

        self.vx = 0.62 * self.vx + 0.38 * dx
        self.vy = 0.62 * self.vy + 0.38 * dy
        self.vw = 0.55 * self.vw + 0.45 * dw
        self.vh = 0.55 * self.vh + 0.45 * dh

        self.x += alpha * dx
        self.y += alpha * dy
        self.w = clamp(self.w + alpha * dw, 8.0, 700.0)
        self.h = clamp(self.h + alpha * dh, 8.0, 700.0)

        self.tlbr_model = xywh_to_xyxy(self.x, self.y, self.w, self.h)
        self.render_tlbr = shift_box(self.tlbr_model, offset_used[0], offset_used[1])

        self.score = float(det["score"])
        if det["score"] >= 0.50:
            self.cls = det["cls"]
            self.color = det["color"]

        self.embedding = l2_normalize(0.75 * self.embedding + 0.25 * np.asarray(det["embedding"], dtype=np.float32))
        self.state = det.get("state", self.state)

        self.time_since_update = 0
        self.hits += 1
        self.last_frame = f
        self.lost = False
        self.recovered_after = duration if was_occluded and duration >= 2 else 0
        self.occluded = False
        self.occluded_start = None
        self.occlusion_event_sent = False


class MOTTracker:
    def __init__(self,
                 algorithm: str,
                 conf_thresh: float = 0.45,
                 iou_thresh: float = 0.25,
                 max_age: int = 12,
                 min_hits: int = 2,
                 appearance_weight: float = 0.30,
                 reid_thresh: float = 0.52,
                 reid_ttl: int = 150,
                 camera_comp: bool = False,
                 fps: int = FPS) -> None:
        self.algorithm = algorithm
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.appearance_weight = float(appearance_weight)
        self.use_appearance = algorithm == "BoT-SORT" and self.appearance_weight > 0.0
        self.reid_thresh = float(reid_thresh)
        self.reid_ttl = int(reid_ttl)
        self.camera_comp = bool(camera_comp)
        self.fps = int(fps)

        self.tracks: List[Track] = []
        self.next_id = 1
        self.gallery: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.review_queue: List[Dict[str, Any]] = []
        self.track_histories: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
        self.max_cost = 0.96

    def add_event(self, frame: int, etype: str, message: str,
                  confidence: Optional[float] = None,
                  track_id: Optional[int] = None,
                  gt_id: Optional[int] = None) -> None:
        self.events.append({
            "frame": int(frame),
            "time_s": round(float(frame) / float(self.fps), 3),
            "type": etype,
            "message": message,
            "confidence": None if confidence is None else round(float(confidence), 3),
            "track_id": track_id,
            "gt_id": gt_id,
        })

    def associate(self, tracks: List[Track], dets: List[Dict[str, Any]],
                  iou_thresh: float, use_appearance: bool
                  ) -> Tuple[List[Tuple[Track, Dict[str, Any]]], List[Track], List[Dict[str, Any]]]:
        if not tracks or not dets:
            return [], tracks, dets

        pairs = []
        for ti, t in enumerate(tracks):
            for di, d in enumerate(dets):
                v = box_iou(t.tlbr_model, d["bbox_model"])
                if v < iou_thresh:
                    continue
                cost = 1.0 - v
                if use_appearance:
                    cost += self.appearance_weight * embedding_distance(t.embedding, d["embedding"])
                if cost <= self.max_cost:
                    pairs.append((cost, ti, di))

        pairs.sort(key=lambda x: x[0])
        matched = []
        used_t = set()
        used_d = set()

        for cost, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            matched.append((tracks[ti], dets[di]))
            used_t.add(ti)
            used_d.add(di)

        rem_tracks = [t for i, t in enumerate(tracks) if i not in used_t]
        rem_dets = [d for i, d in enumerate(dets) if i not in used_d]
        return matched, rem_tracks, rem_dets

    def add_to_gallery(self, t: Track, f: int) -> None:
        if t.hits < 2:
            return
        self.gallery.append({
            "track_id": t.id,
            "cls": t.cls,
            "color": t.color,
            "embedding": t.embedding.copy(),
            "bbox_model": list(t.tlbr_model),
            "frame": t.last_frame,
            "hits": t.hits,
            "confirmed": t.confirmed,
            "created_frame": t.created_frame,
            "state": t.state,
        })

    def try_reactivate(self, det: Dict[str, Any], f: int, offset_used: Tuple[int, int]) -> bool:
        if det["score"] < 0.22 or not self.gallery:
            return False

        best_idx = -1
        best_score = 0.0
        best_g = None

        for gi, g in enumerate(self.gallery):
            time_gap = max(1, f - g["frame"])
            if time_gap > self.reid_ttl:
                continue

            # Class compatibility
            a = g["cls"]
            b = det["cls"]
            compatible = (a == b) or (a in ("vehicle", "forklift") and b in ("vehicle", "forklift")) or \
                         (a in ("container", "shelf_item") and b in ("container", "shelf_item"))
            if not compatible:
                continue

            emb_dist = embedding_distance(g["embedding"], det["embedding"])
            color_dist = 0.0 if g["color"] == det["color"] else 0.35

            gc = center_of_box(g["bbox_model"])
            dc = center_of_box(det["bbox_model"])
            loc_dist = math.hypot(gc[0] - dc[0], gc[1] - dc[1])
            max_move = 7.5 * time_gap + 55.0
            loc_cost = min(1.0, loc_dist / max_move)

            score = 1.0 - (0.55 * emb_dist + 0.25 * color_dist + 0.20 * loc_cost)
            if score > best_score:
                best_score = score
                best_idx = gi
                best_g = g

        if best_g is None:
            return False

        if best_score >= self.reid_thresh:
            t = Track(best_g["track_id"], det, f, offset_used)
            t.hits = int(best_g["hits"]) + 1
            t.confirmed = bool(best_g["confirmed"]) or t.hits >= self.min_hits
            t.created_frame = int(best_g["created_frame"])
            t.reid_conf = float(best_score)
            self.tracks.append(t)
            del self.gallery[best_idx]

            self.add_event(
                f, "reid_match",
                f"Re-ID reconnected detection to Track #{t.id} (confidence {best_score:.2f})",
                confidence=best_score, track_id=t.id
            )
            return True

        if best_score >= self.reid_thresh - 0.12:
            duplicate = any(
                r["track_id"] == best_g["track_id"] and f - r["frame"] < 18
                for r in self.review_queue
            )
            if not duplicate:
                self.add_event(
                    f, "reid_uncertain",
                    f"Uncertain Re-ID candidate for Track #{best_g['track_id']} (score {best_score:.2f})",
                    confidence=best_score, track_id=best_g["track_id"]
                )
                self.review_queue.append({
                    "frame": f,
                    "time_s": round(f / float(self.fps), 2),
                    "track_id": best_g["track_id"],
                    "confidence": round(float(best_score), 3),
                    "class": det["cls"],
                    "color": det["color"],
                    "status": "pending",
                })
        return False

    def process(self, dets: List[Dict[str, Any]], f: int, offset_used: Tuple[int, int]) -> List[Dict[str, Any]]:
        for d in dets:
            d["bbox_model"] = shift_box(d["bbox"], -offset_used[0], -offset_used[1])

        high = [d for d in dets if d["score"] >= self.conf_thresh]
        low = [d for d in dets if d["score"] < self.conf_thresh]

        for t in self.tracks:
            t.predict()

        for t in self.tracks:
            if t.time_since_update == 1 and t.confirmed and not t.lost and not t.occlusion_event_sent:
                t.occlusion_event_sent = True
                self.add_event(f, "occlusion_start", f"Track #{t.id} missing / occlusion start", track_id=t.id)

        candidates = [t for t in self.tracks if not t.deleted]
        matches, rem_tracks, _ = self.associate(candidates, high, self.iou_thresh, self.use_appearance)
        for t, d in matches:
            t.update(d, f, offset_used)

        low_candidates = [t for t in rem_tracks if t.time_since_update > 0]
        matches_low, _, _ = self.associate(
            low_candidates, low,
            max(0.05, self.iou_thresh - 0.10),
            False
        )
        for t, d in matches_low:
            t.update(d, f, offset_used)

        matched_det_ids = set(id(d) for _, d in matches)
        matched_det_ids.update(id(d) for _, d in matches_low)
        unmatched_dets = [d for d in dets if id(d) not in matched_det_ids]

        still_unmatched = []
        for d in unmatched_dets:
            if not self.try_reactivate(d, f, offset_used):
                still_unmatched.append(d)

        for d in still_unmatched:
            if d["score"] >= self.conf_thresh or (d["score"] >= 0.30 and self.algorithm == "BoT-SORT"):
                t = Track(self.next_id, d, f, offset_used)
                self.next_id += 1
                self.tracks.append(t)
                self.add_event(
                    f, "track_created",
                    f"Track #{t.id} created ({t.cls}, {t.color})",
                    confidence=t.score, track_id=t.id
                )

        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True

            if t.recovered_after >= 2:
                self.add_event(
                    f, "occlusion_recovery",
                    f"Track #{t.id} recovered after {t.recovered_after} frames",
                    confidence=t.score, track_id=t.id
                )
                t.recovered_after = 0

            if t.time_since_update > self.max_age and not t.lost:
                t.lost = True
                self.add_to_gallery(t, f)
                self.add_event(f, "track_lost", f"Track #{t.id} lost", track_id=t.id)
                t.deleted = True

        self.gallery = [g for g in self.gallery if f - g["frame"] <= self.reid_ttl]
        self.tracks = [t for t in self.tracks if not t.deleted]

        # Keep rendered boxes aligned with current camera offset, including occluded tracks.
        for t in self.tracks:
            if not t.deleted:
                t.render_tlbr = shift_box(t.tlbr_model, offset_used[0], offset_used[1])

        record = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                bbox_tuple = (int(t.render_tlbr[0]), int(t.render_tlbr[1]), int(t.render_tlbr[2]), int(t.render_tlbr[3]))
                self.track_histories[t.id].append((f, bbox_tuple))
                record.append({
                    "track_id": t.id,
                    "bbox": list(bbox_tuple),
                    "cls": t.cls,
                    "color": t.color,
                    "score": round(float(t.score), 3),
                    "state": t.state,
                    "occluded": bool(t.occluded),
                    "confirmed": bool(t.confirmed),
                    "reid_conf": None if t.reid_conf is None else round(float(t.reid_conf), 3),
                    "age": int(t.age),
                    "time_since_update": int(t.time_since_update),
                })

        return record


# ------------------------------------------------------------------------------
# Annotation / drawing
# ------------------------------------------------------------------------------
def draw_annotations(frame: np.ndarray,
                     rec: List[Dict[str, Any]],
                     f: int,
                     params: Dict[str, Any],
                     offset_used: Tuple[int, int],
                     est_off: Tuple[int, int],
                     abs_off: Tuple[int, int],
                     gt: List[Dict[str, Any]],
                     histories: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]],
                     events: List[Dict[str, Any]],
                     query: Optional[str] = None,
                     show_trails: bool = True,
                     show_gt: bool = False,
                     trail_length: int = 22) -> np.ndarray:

    q = params.get("query", "") if query is None else query
    colors, classes = parse_query(q)

    if show_trails:
        for item in rec:
            full_hist = histories.get(item["track_id"], [])
            hist = [(fr, bx) for fr, bx in full_hist if fr <= f][-trail_length:]
            prev = None
            col = id_color(item["track_id"])
            for _, hbox in hist:
                c = center_of_box(list(hbox))
                if prev is not None:
                    cv2.line(frame, prev, c, col, 1, cv2.LINE_AA)
                prev = c

    if show_gt:
        for g in gt:
            cv2.rectangle(frame, (g["bbox"][0], g["bbox"][1]), (g["bbox"][2], g["bbox"][3]), (0, 255, 0), 1, cv2.LINE_AA)

    counts = defaultdict(int)
    for item in rec:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        col = id_color(item["track_id"])
        th = 2 if item.get("confirmed") else 1

        if item.get("occluded"):
            draw_dashed_rectangle(frame, [x1, y1, x2, y2], (0, 0, 255), max(1, th), 7)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)

        label = f"#{item['track_id']} {item['cls']}"
        if item.get("state") not in (None, "normal"):
            label += f" [{item['state']}]"
        if item.get("reid_conf"):
            label += f" rid:{item['reid_conf']:.2f}"

        cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{item['score']:.2f}", (x1, min(HEIGHT - 4, y2 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

        counts[item["cls"]] += 1

        qs = query_match_score(item["cls"], item["color"], colors, classes)
        if qs >= 0.60:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 120), 3, cv2.LINE_AA)
            cv2.putText(frame, "QUERY", (x1 - 3, max(14, y1 - 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

    draw_transparent_rect(frame, (0, 0), (WIDTH, 96), (18, 18, 28), 0.58)
    count_txt = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) if counts else "none"
    lines = [
        f"Frame {f + 1}/{params.get('total_frames', '?')}  t={f / float(FPS):.1f}s  Tracker={params.get('algorithm', '?')}",
        f"Active IDs: {len(rec)}  GT: {len(gt)}  Counts: {count_txt}",
        f"CamComp={'ON' if params.get('camera_comp') else 'OFF'}  est=({est_off[0]},{est_off[1]})  abs=({abs_off[0]},{abs_off[1]})",
        f"Open-vocab query: {q if q else '-'}"
    ]
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        y += 18

    flags = []
    if max(abs(abs_off[0]), abs(abs_off[1])) > 18:
        flags.append("CAMERA MOTION")
    occ_n = sum(1 for item in rec if item.get("occluded"))
    if occ_n:
        flags.append(f"OCCLUSION x{occ_n}")
    if any(item.get("reid_conf") for item in rec):
        flags.append("RE-ID")
    if any(e["type"] == "state_change" and 0 <= f - e["frame"] < 10 for e in events):
        flags.append("STATE CHANGE")
    if any(e["type"] == "camera_handoff" and 0 <= f - e["frame"] < 10 for e in events):
        flags.append("HANDOFF")
    if any(e["type"] in ("item_in", "item_out") and 0 <= f - e["frame"] < 10 for e in events):
        flags.append("INVENTORY EVENT")

    if flags:
        txt = " | ".join(flags)
        tw = max(180, len(txt) * 9)
        draw_transparent_rect(frame, (WIDTH - tw - 8, HEIGHT - 34), (WIDTH - 4, HEIGHT - 5), (0, 0, 0), 0.55)
        cv2.putText(frame, txt, (WIDTH - tw, HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    recent = [e for e in events if e["frame"] <= f][-3:]
    if recent:
        draw_transparent_rect(frame, (0, HEIGHT - 58), (430, HEIGHT - 4), (0, 0, 0), 0.45)
        yy = HEIGHT - 43
        for e in recent:
            msg = f"t={e['time_s']:.1f}s {e['type']}: {e['message'][:58]}"
            cv2.putText(frame, msg, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
            yy += 15

    return frame


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def compute_metrics(frame_records: List[List[Dict[str, Any]]],
                    gt_records: List[List[Dict[str, Any]]],
                    total_frames: int,
                    events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    events = events or []

    tp = fp = fn = 0
    iou_sum = 0.0
    assoc = defaultdict(int)
    gt_frames = defaultdict(int)
    pred_frames = defaultdict(int)
    total_gt = 0
    total_pred = 0
    last_map: Dict[int, int] = {}
    id_switches = 0

    for rec, gts in zip(frame_records, gt_records):
        for g in gts:
            gt_frames[g["gt_id"]] += 1
            total_gt += 1
        for r in rec:
            pred_frames[r["track_id"]] += 1
            total_pred += 1

        pairs = []
        for gi, g in enumerate(gts):
            for ri, r in enumerate(rec):
                v = box_iou(g["bbox"], r["bbox"])
                if v >= 0.50:
                    pairs.append((v, gi, ri))
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_g = set()
        used_r = set()
        frame_map = {}

        for v, gi, ri in pairs:
            if gi in used_g or ri in used_r:
                continue
            used_g.add(gi)
            used_r.add(ri)
            tp += 1
            iou_sum += v
            gid = gts[gi]["gt_id"]
            tid = rec[ri]["track_id"]
            assoc[(gid, tid)] += 1
            frame_map[gid] = tid

        fn += len(gts) - len(used_g)
        fp += len(rec) - len(used_r)

        for gid, tid in frame_map.items():
            if gid in last_map and last_map[gid] != tid:
                id_switches += 1
            last_map[gid] = tid

    det_a = tp / max(1.0, tp + 0.5 * fp + 0.5 * fn)
    ass_a = iou_sum / max(1.0, float(tp))
    hota = math.sqrt(max(0.0, det_a) * max(0.0, ass_a))
    mota = 1.0 - (fn + fp + id_switches) / max(1.0, float(total_gt))

    gt_to_best: Dict[int, Tuple[int, int]] = {}
    for (gid, tid), cnt in assoc.items():
        if gid not in gt_to_best or cnt > gt_to_best[gid][1]:
            gt_to_best[gid] = (tid, cnt)

    idtp = sum(cnt for _, cnt in gt_to_best.values())
    idfn = total_gt - idtp
    idfp = total_pred - idtp
    idf1 = (2.0 * idtp) / max(1.0, 2.0 * idtp + idfp + idfn)

    occl_starts = sum(1 for e in events if e["type"] == "occlusion_start")
    occl_recov = sum(1 for e in events if e["type"] == "occlusion_recovery")
    recovery_rate = occl_recov / max(1.0, float(occl_starts)) if occl_starts else 1.0

    return {
        "HOTA_approx": round(float(hota), 4),
        "IDF1": round(float(idf1), 4),
        "MOTA_approx": round(float(mota), 4),
        "DetA": round(float(det_a), 4),
        "AssA": round(float(ass_a), 4),
        "ID_switches": int(id_switches),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "total_gt_frames": int(total_gt),
        "total_pred_frames": int(total_pred),
        "occlusion_starts": int(occl_starts),
        "occlusion_recoveries": int(occl_recov),
        "recovery_rate": round(float(recovery_rate), 4),
        "frames": int(total_frames),
    }


def benchmark_from_dets(dets_records: List[List[Dict[str, Any]]],
                        est_offsets: List[Tuple[int, int]],
                        gt_records: List[List[Dict[str, Any]]],
                        params: Dict[str, Any],
                        total_frames: int) -> Dict[str, Any]:
    out = {}
    for alg in ("ByteTrack", "BoT-SORT"):
        appearance = float(params["appearance_weight"]) if alg == "BoT-SORT" else 0.0
        camera_comp = bool(params["camera_comp"])
        tr = MOTTracker(
            algorithm=alg,
            conf_thresh=float(params["confidence"]),
            iou_thresh=float(params["iou"]),
            max_age=int(params["max_age"]),
            min_hits=2,
            appearance_weight=appearance,
            reid_thresh=float(params["reid_threshold"]),
            reid_ttl=180,
            camera_comp=camera_comp,
            fps=FPS,
        )
        recs = []
        t0 = time.perf_counter()
        for f, dets in enumerate(dets_records):
            offset = est_offsets[f] if camera_comp else (0, 0)
            rec = tr.process([d.copy() for d in dets], f, offset)
            recs.append(rec)
        latency_ms = (time.perf_counter() - t0) / max(1, total_frames) * 1000.0
        m = compute_metrics(recs, gt_records, total_frames, tr.events)
        m["latency_ms"] = round(float(latency_ms), 3)
        out[alg] = {"metrics": m, "events": tr.events}
    return out


# ------------------------------------------------------------------------------
# Video writer helper
# ------------------------------------------------------------------------------
def _open_video_writer(path: str, fourcc: str, fps: int, size: Tuple[int, int]):
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    except Exception:
        pass
    return None


# ------------------------------------------------------------------------------
# Full simulation
# ------------------------------------------------------------------------------
def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    total = max(30, int(float(params["duration"]) * FPS))
    p = dict(params)
    p["total_frames"] = total

    objects = make_objects(p["scenario"], total)
    bg = make_background(WIDTH, HEIGHT, p["scenario"])

    tracker = MOTTracker(
        algorithm=p["algorithm"],
        conf_thresh=p["confidence"],
        iou_thresh=p["iou"],
        max_age=p["max_age"],
        min_hits=2,
        appearance_weight=p["appearance_weight"] if p["algorithm"] == "BoT-SORT" else 0.0,
        reid_thresh=p["reid_threshold"],
        reid_ttl=180,
        camera_comp=p["camera_comp"],
        fps=FPS,
    )

    estimator = CameraMotionEstimator()

    video_bytes = b""
    mime = None
    writer = None
    tmp_path = None

    for suffix, fourcc, candidate_mime in (
        (".mp4", "mp4v", "video/mp4"),
        (".avi", "MJPG", "video/x-msvideo"),
    ):
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.close()
            tmp_path = tmp.name
        except Exception:
            tmp_path = None
            continue

        writer = _open_video_writer(tmp_path, fourcc, FPS, (WIDTH, HEIGHT))
        if writer is not None:
            mime = candidate_mime
            break

        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            tmp_path = None

    dets_records: List[List[Dict[str, Any]]] = []
    est_offsets: List[Tuple[int, int]] = []
    abs_offsets: List[Tuple[int, int]] = []
    gt_records: List[List[Dict[str, Any]]] = []
    frame_records: List[List[Dict[str, Any]]] = []
    series = defaultdict(list)
    latencies: List[float] = []
    gif_frames = []

    prev_states: Dict[int, str] = {}
    prev_visible: Dict[int, bool] = {}
    prev_centers: Dict[int, float] = {}
    prev_possession: Optional[int] = None

    start_time = time.time()

    for f in range(total):
        t0 = time.perf_counter()

        abs_off = absolute_camera_offset(f, p["scenario"], total)
        raw = render_raw(objects, f, abs_off, bg, p["scenario"])
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        est_off = estimator.estimate(gray)

        offset_main = est_off if p["camera_comp"] else (0, 0)
        dets = simulate_detections(raw, objects, f, abs_off, p)

        gt = []
        for obj in objects:
            pose = object_pose(obj, f)
            if pose is None:
                continue
            box = xywh_to_xyxy(pose["cx"] + abs_off[0], pose["cy"] + abs_off[1], pose["w"], pose["h"])
            gt.append({
                "gt_id": obj["id"],
                "bbox": box,
                "cls": obj["cls"],
                "color": obj["color"],
                "state": pose["state"],
            })

            # State transition events
            old_state = prev_states.get(obj["id"])
            if old_state is not None and old_state != pose["state"]:
                tracker.add_event(
                    f, "state_change",
                    f"{obj['cls']} #{obj['id']} state: {old_state} -> {pose['state']}",
                    confidence=0.90, gt_id=obj["id"]
                )
            prev_states[obj["id"]] = pose["state"]

            # Retail item in/out events
            if obj["cls"] == "shelf_item" and f > 0:
                was_visible = prev_visible.get(obj["id"], False)
                if not was_visible:
                    tracker.add_event(f, "item_in", f"Shelf item #{obj['id']} entered zone", confidence=0.88, gt_id=obj["id"])
                prev_visible[obj["id"]] = True
            elif obj["cls"] == "shelf_item":
                prev_visible.setdefault(obj["id"], True)

            # Multi-camera handoff hypothesis
            if p["scenario"] in ("multicamera", "integrated"):
                old_cx = prev_centers.get(obj["id"])
                new_cx = pose["cx"]
                if old_cx is not None:
                    if (old_cx < WIDTH / 2.0 <= new_cx) or (old_cx > WIDTH / 2.0 >= new_cx):
                        tracker.add_event(
                            f, "camera_handoff",
                            f"Object #{obj['id']} crossed Camera A/B boundary; handoff hypothesis logged",
                            confidence=0.78, gt_id=obj["id"]
                        )
                prev_centers[obj["id"]] = new_cx

        # Disappearance events for shelf items
        if f > 0:
            for obj in objects:
                if obj["cls"] != "shelf_item":
                    continue
                visible_now = any(g["gt_id"] == obj["id"] for g in gt)
                was_visible = prev_visible.get(obj["id"], False)
                if was_visible and not visible_now:
                    tracker.add_event(f, "item_out", f"Shelf item #{obj['id']} removed from zone", confidence=0.86, gt_id=obj["id"])
                prev_visible[obj["id"]] = visible_now

        # Sports possession candidate
        ball = next((g for g in gt if g["cls"] == "ball"), None)
        if ball is not None:
            persons = [g for g in gt if g["cls"] == "person"]
            best = None
            best_d = 1e9
            bc = center_of_box(ball["bbox"])
            for pr in persons:
                pc = center_of_box(pr["bbox"])
                d = math.hypot(bc[0] - pc[0], bc[1] - pc[1])
                if d < best_d:
                    best_d = d
                    best = pr
            if best is not None and best_d < 130.0:
                if prev_possession is None or prev_possession != best["gt_id"]:
                    tracker.add_event(
                        f, "possession_change",
                        f"Ball possession candidate switched to player #{best['gt_id']}",
                        confidence=round(float(clamp(1.0 - best_d / 130.0, 0.2, 0.95)), 3),
                        gt_id=best["gt_id"]
                    )
                prev_possession = best["gt_id"]
            else:
                prev_possession = None

        rec = tracker.process([d.copy() for d in dets], f, offset_main)

        annotated = draw_annotations(
            raw.copy(), rec, f, p, offset_main, est_off, abs_off,
            gt, tracker.track_histories, tracker.events,
            query=p.get("query", ""), show_trails=True, show_gt=False, trail_length=22
        )

        if writer is not None:
            writer.write(np.ascontiguousarray(annotated, dtype=np.uint8))

        if PIL_AVAILABLE:
            try:
                rgb = cv2.cvtColor(np.ascontiguousarray(annotated), cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize((GIF_WIDTH, GIF_HEIGHT), PIL_LANCZOS)
                gif_frames.append(pil)
            except Exception:
                pass

        frame_records.append(rec)
        gt_records.append(gt)
        dets_records.append(dets)
        est_offsets.append(est_off)
        abs_offsets.append(abs_off)

        series["frame"].append(f)
        series["active_tracks"].append(len(rec))
        series["detections"].append(len(dets))
        series["gt_objects"].append(len(gt))

        latencies.append(time.perf_counter() - t0)

    if writer is not None:
        writer.release()
        if tmp_path and os.path.exists(tmp_path):
            try:
                with open(tmp_path, "rb") as fh:
                    video_bytes = fh.read()
            except Exception:
                video_bytes = b""
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    gif_bytes = b""
    if gif_frames:
        try:
            buf = io.BytesIO()
            gif_frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=gif_frames[1:],
                duration=int(round(1000.0 / FPS)),
                loop=0,
                optimize=False,
            )
            gif_bytes = buf.getvalue()
        except Exception:
            gif_bytes = b""

    elapsed = max(1e-6, time.time() - start_time)
    metrics = compute_metrics(frame_records, gt_records, total, tracker.events)
    metrics["end_to_end_fps"] = round(float(total / elapsed), 3)
    metrics["avg_latency_ms"] = round(float(np.mean(latencies) * 1000.0), 3) if latencies else 0.0

    benchmark_results = None
    if p.get("benchmark", False):
        try:
            benchmark_results = benchmark_from_dets(dets_records, est_offsets, gt_records, p, total)
        except Exception:
            benchmark_results = None

    return {
        "video_bytes": video_bytes,
        "gif_bytes": gif_bytes,
        "mime": mime,
        "objects": objects,
        "bg": bg,
        "abs_offsets": abs_offsets,
        "est_offsets": est_offsets,
        "frame_records": frame_records,
        "gt_records": gt_records,
        "dets_records": dets_records,
        "track_histories": dict(tracker.track_histories),
        "events": tracker.events,
        "review_queue": tracker.review_queue,
        "metrics": metrics,
        "benchmark": benchmark_results,
        "series": dict(series),
        "params": p,
        "total_frames": total,
    }


def render_explorer_frame(result: Dict[str, Any], f: int, query: str,
                          show_trails: bool, show_gt: bool) -> np.ndarray:
    p = result["params"]
    abs_off = result["abs_offsets"][f]
    est_off = result["est_offsets"][f]
    offset_used = est_off if p["camera_comp"] else (0, 0)

    raw = render_raw(result["objects"], f, abs_off, result["bg"], p["scenario"])
    ann = draw_annotations(
        raw.copy(),
        result["frame_records"][f],
        f,
        p,
        offset_used,
        est_off,
        abs_off,
        result["gt_records"][f],
        result["track_histories"],
        result["events"],
        query=query,
        show_trails=show_trails,
        show_gt=show_gt,
        trail_length=28,
    )
    return cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)


def events_to_csv(events: List[Dict[str, Any]]) -> bytes:
    fields = ["frame", "time_s", "type", "message", "confidence", "track_id", "gt_id", "severity"]
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=fields)
    w.writeheader()
    for e in events:
        sev = "warning" if e["type"] in (
            "track_lost", "reid_uncertain", "state_change", "item_out", "occlusion_start"
        ) else "info"
        row = {
            "frame": e.get("frame", ""),
            "time_s": e.get("time_s", ""),
            "type": e.get("type", ""),
            "message": e.get("message", ""),
            "confidence": "" if e.get("confidence") is None else e.get("confidence"),
            "track_id": "" if e.get("track_id") is None else e.get("track_id"),
            "gt_id": "" if e.get("gt_id") is None else e.get("gt_id"),
            "severity": sev,
        }
        w.writerow(row)
    return sio.getvalue().encode("utf-8")


def build_track_summary(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    score_sum = defaultdict(float)
    score_cnt = defaultdict(int)
    attrs = {}
    reid_max = defaultdict(float)

    for rec in result["frame_records"]:
        for r in rec:
            tid = r["track_id"]
            score_sum[tid] += float(r["score"])
            score_cnt[tid] += 1
            if tid not in attrs:
                attrs[tid] = {"cls": r["cls"], "color": r["color"]}
            if r.get("reid_conf"):
                reid_max[tid] = max(reid_max[tid], float(r["reid_conf"]))

    rows = []
    for tid, hist in result["track_histories"].items():
        if not hist:
            continue
        rows.append({
            "track_id": tid,
            "class": attrs.get(tid, {}).get("cls", ""),
            "color": attrs.get(tid, {}).get("color", ""),
            "first_frame": hist[0][0],
            "last_frame": hist[-1][0],
            "frames_observed": len(hist),
            "avg_score": round(score_sum[tid] / max(1, score_cnt[tid]), 3),
            "max_reid_conf": round(reid_max.get(tid, 0.0), 3),
        })
    rows.sort(key=lambda x: x["track_id"])
    return rows


def track_summary_to_csv(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b"track_id,class,color,first_frame,last_frame,frames_observed,avg_score,max_reid_conf\n"
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return sio.getvalue().encode("utf-8")


# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------
if "applied_params" not in st.session_state:
    st.session_state.applied_params = DEFAULT_PARAMS.copy()
if "review_decisions" not in st.session_state:
    st.session_state.review_decisions = {}
if "result_hash" not in st.session_state:
    st.session_state.result_hash = None
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

with st.sidebar:
    st.title("🎛️ Mission Controls")
    st.caption("No API key. Offline synthetic field demo.")

    ap = st.session_state.applied_params

    with st.form("settings_form"):
        scenario = st.selectbox(
            "Scenario",
            SCENARIOS,
            index=SCENARIOS.index(ap["scenario"]) if ap["scenario"] in SCENARIOS else 0
        )
        algorithm = st.radio(
            "Tracker",
            ["ByteTrack", "BoT-SORT"],
            index=0 if ap["algorithm"] == "ByteTrack" else 1,
            horizontal=True
        )
        confidence = st.slider("Detection confidence threshold", 0.20, 0.80, float(ap["confidence"]), 0.05)
        iou_value = st.slider("IoU match threshold", 0.05, 0.50, float(ap["iou"]), 0.05)
        max_age = st.slider("Max lost frames", 5, 30, int(ap["max_age"]))
        appearance = st.slider("BoT-SORT appearance weight", 0.00, 0.60, float(ap["appearance_weight"]), 0.05)
        reid_threshold = st.slider("Re-ID confidence threshold", 0.30, 0.85, float(ap["reid_threshold"]), 0.05)
        camera_comp = st.checkbox("Camera-motion compensation", bool(ap["camera_comp"]))
        det_noise = st.slider("Detector noise", 0.00, 1.00, float(ap["det_noise"]), 0.05)
        duration = st.slider("Demo duration (seconds)", 6, 20, int(ap["duration"]))
        query = st.text_input("Open-vocabulary query", ap["query"])
        benchmark = st.checkbox("Run ByteTrack vs BoT-SORT benchmark", bool(ap["benchmark"]))
        seed = st.number_input("Random seed", value=int(ap["seed"]), step=1, format="%d")

        submitted = st.form_submit_button("Apply & Regenerate Live Demo", use_container_width=True)

    if submitted:
        st.session_state.applied_params = {
            "scenario": scenario,
            "algorithm": algorithm,
            "confidence": float(confidence),
            "iou": float(iou_value),
            "max_age": int(max_age),
            "appearance_weight": float(appearance),
            "reid_threshold": float(reid_threshold),
            "camera_comp": bool(camera_comp),
            "det_noise": float(det_noise),
            "duration": int(duration),
            "query": str(query),
            "benchmark": bool(benchmark),
            "seed": int(seed),
        }
        st.session_state.result_hash = None
        st.session_state.review_decisions = {}

    st.divider()
    st.markdown(
        "**Tips**\n"
        "- Try query: `red backpack`, `yellow forklift`, `dog`, `door open`.\n"
        "- Switch scenario to isolate occlusion, re-ID, camera motion, retail, sports, wildlife.\n"
        "- BoT-SORT uses appearance + re-ID gallery; ByteTrack is primarily IoU-based."
    )

params = st.session_state.applied_params
params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()

if st.session_state.result_hash != params_hash or st.session_state.result is None:
    with st.spinner("Generating live demo preview, tracks, events, and metrics..."):
        try:
            st.session_state.result = run_simulation(params)
            st.session_state.error = None
            st.session_state.result_hash = params_hash
        except Exception:
            st.session_state.result = None
            st.session_state.error = traceback.format_exc()

if st.session_state.result is None:
    st.error("Simulation failed. See details below.")
    st.code(st.session_state.error or "Unknown error", language="text")
    st.stop()

result = st.session_state.result
m = result["metrics"]
p = result["params"]

st.title("🎥 Autonomous Multi-Object Tracking Engine")
st.caption(
    "Detector → tracker → trajectory store → live dashboard. "
    "This demo auto-generates a realistic synthetic field video, preserves track IDs, "
    "estimates motion, handles occlusion/re-ID/camera motion, and exports annotated video + event log."
)

tab_live, tab_explorer, tab_metrics, tab_review, tab_modules = st.tabs([
    "Live Demo",
    "Track Explorer",
    "Metrics & Events",
    "Re-ID Review",
    "Modules",
])

with tab_live:
    st.subheader("Live animated tracking preview")
    st.caption("This preview is generated automatically from the tracked video frames.")

    preview_shown = False

    if result.get("gif_bytes"):
        b64gif = base64.b64encode(result["gif_bytes"]).decode("utf-8")
        st.markdown(
            f'<img src="data:image/gif;base64,{b64gif}" style="width:100%; border-radius:8px;" />',
            unsafe_allow_html=True,
        )
        preview_shown = True
    elif result.get("video_bytes"):
        if result.get("mime") == "video/mp4":
            b64 = base64.b64encode(result["video_bytes"]).decode("utf-8")
            st.markdown(
                f"""
                <video width="100%" autoplay muted loop controls>
                  <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                  Your browser does not support embedded video.
                </video>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.video(result["video_bytes"], format=result["mime"], loop=True, autoplay=True, muted=True)
        preview_shown = True

    if not preview_shown:
        st.warning("Video/GIF backend unavailable in this environment. Showing final annotated frame instead.")
        final_img = render_explorer_frame(result, result["total_frames"] - 1, p["query"], True, False)
        st.image(final_img, use_container_width=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("HOTA (approx)", f"{m['HOTA_approx']:.3f}")
    c2.metric("IDF1", f"{m['IDF1']:.3f}")
    c3.metric("ID switches", f"{m['ID_switches']}")
    c4.metric("End-to-end FPS", f"{m['end_to_end_fps']:.1f}")
    c5.metric("Avg latency", f"{m['avg_latency_ms']:.1f} ms")
    c6.metric("Active IDs (last)", f"{len(result['frame_records'][-1])}")

    cc1, cc2 = st.columns([3, 1])
    with cc1:
        st.markdown("**Latest events**")
        recent = list(reversed(result["events"]))[:12]
        if recent:
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.info("No events logged.")
    with cc2:
        if result.get("video_bytes"):
            video_ext = "mp4" if result.get("mime") == "video/mp4" else "avi"
            st.download_button(
                f"Download annotated {video_ext.upper()}",
                data=result["video_bytes"],
                file_name=f"mot_live_demo.{video_ext}",
                mime=result["mime"],
                use_container_width=True,
            )

        if result.get("gif_bytes"):
            st.download_button(
                "Download animated GIF preview",
                data=result["gif_bytes"],
                file_name="mot_live_preview.gif",
                mime="image/gif",
                use_container_width=True,
            )

        st.download_button(
            "Download event log CSV",
            data=events_to_csv(result["events"]),
            file_name="mot_event_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tab_explorer:
    st.subheader("Interactive frame explorer")
    fc1, fc2, fc3 = st.columns([3, 1, 1])
    with fc1:
        frame_idx = st.slider("Frame", 0, result["total_frames"] - 1, result["total_frames"] - 1)
    with fc2:
        live_query = st.text_input("Live query preview", p["query"])
    with fc3:
        show_trails = st.checkbox("Trails", True)
        show_gt = st.checkbox("Ground truth", False)

    img_rgb = render_explorer_frame(result, frame_idx, live_query, show_trails, show_gt)
    st.image(img_rgb, use_container_width=True)

    rec = result["frame_records"][frame_idx]
    st.markdown("**Tracks in selected frame**")
    if rec:
        st.dataframe(rec, use_container_width=True, hide_index=True)
    else:
        st.info("No active tracks in this frame.")

    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button(
            "Download current frame PNG",
            data=buf.tobytes(),
            file_name=f"mot_frame_{frame_idx:04d}.png",
            mime="image/png",
        )

with tab_metrics:
    st.subheader("MOT-style evaluation")
    st.caption(
        "Metrics are computed against synthetic ground truth. "
        "HOTA is an approximate lightweight implementation suitable for demo monitoring."
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("DetA", f"{m['DetA']:.3f}")
    mc2.metric("AssA", f"{m['AssA']:.3f}")
    mc3.metric("MOTA (approx)", f"{m['MOTA_approx']:.3f}")
    mc4.metric("Recovery rate", f"{m['recovery_rate']:.3f}")

    st.line_chart({
        "Frame": result["series"]["frame"],
        "Active tracks": result["series"]["active_tracks"],
        "Detections": result["series"]["detections"],
        "GT objects": result["series"]["gt_objects"],
    })

    st.markdown("**ByteTrack vs BoT-SORT benchmark**")
    if not result.get("benchmark"):
        if st.button("Run A/B benchmark now"):
            with st.spinner("Benchmarking trackers..."):
                try:
                    st.session_state.result["benchmark"] = benchmark_from_dets(
                        result["dets_records"],
                        result["est_offsets"],
                        result["gt_records"],
                        result["params"],
                        result["total_frames"],
                    )
                except Exception:
                    st.session_state.result["benchmark"] = None
                    st.error("Benchmark failed, but the main demo is still available.")
            st.rerun()
    else:
        rows = []
        for alg, payload in result["benchmark"].items():
            bm = payload["metrics"]
            rows.append({
                "Algorithm": alg,
                "HOTA": bm["HOTA_approx"],
                "IDF1": bm["IDF1"],
                "MOTA": bm["MOTA_approx"],
                "ID switches": bm["ID_switches"],
                "FP": bm["FP"],
                "FN": bm["FN"],
                "Recovery rate": bm["recovery_rate"],
                "Latency ms": bm.get("latency_ms", ""),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("**Track summary**")
    summary = build_track_summary(result)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    dc1, dc2 = st.columns(2)
    dc1.download_button(
        "Download track summary CSV",
        data=track_summary_to_csv(summary),
        file_name="mot_track_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dc2.download_button(
        "Download full event log CSV",
        data=events_to_csv(result["events"]),
        file_name="mot_full_event_log.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab_review:
    st.subheader("Uncertain Re-ID matches for human review")
    st.caption(
        "The system proposes matches with confidence. "
        "Low-confidence matches are not automatically treated as guaranteed identity."
    )

    review = result.get("review_queue", [])
    if not review:
        st.success("No uncertain Re-ID proposals were generated for this run.")
    else:
        for idx, item in enumerate(review):
            cols = st.columns([5, 1, 1, 1])
            decision = st.session_state.review_decisions.get(idx, "pending")
            cols[0].markdown(
                f"**Frame {item['frame']}** · Track #{item['track_id']} · "
                f"{item['class']}/{item['color']} · confidence `{item['confidence']:.2f}`"
            )
            cols[1].metric("Status", decision)
            if cols[2].button("Accept", key=f"accept_{idx}", use_container_width=True):
                st.session_state.review_decisions[idx] = "accepted"
                st.rerun()
            if cols[3].button("Reject", key=f"reject_{idx}", use_container_width=True):
                st.session_state.review_decisions[idx] = "rejected"
                st.rerun()

with tab_modules:
    st.subheader("How this app maps to the field mission")
    st.markdown(
        """
        ### Core pipeline
        - **Detector:** simulated detector with noise, misses, occlusion penalties, confidence scores, and appearance embeddings extracted from image crops.
        - **Tracker:** ByteTrack-style and BoT-SORT-style association using IoU, motion prediction, appearance cost, low-score detection reuse, and track lifecycle management.
        - **Trajectory store:** per-track histories, active track records, event log, downloadable track summary.
        - **Live dashboard:** automatic animated preview, frame explorer, metrics, event ticker, downloads.

        ### Requested capabilities represented in the demo
        - **Stable IDs across frames:** confirmed tracks, tentative tracks, occlusion handling, lost-track gallery.
        - **Motion estimation:** constant-velocity prediction, velocity damping, smoothed bbox updates.
        - **Scene changes:** camera-motion flag, state-change events, handoff events, inventory events, possession events.
        - **ByteTrack vs BoT-SORT:** selectable tracker and optional A/B benchmark table.
        - **Export annotated video:** downloadable video file generated automatically when backend allows.
        - **Long-term re-identification:** appearance embeddings + time/location constraints + confidence threshold + human review queue.
        - **Crowded occlusion recovery:** occlusion state machine, dashed occlusion boxes, recovery events, recovery-rate metric.
        - **Camera-motion compensation:** sparse optical-flow global motion estimator, compensated tracking coordinate frame, static-camera baseline via toggle.
        - **Open-vocabulary tracking:** text query parser for color/class terms, query highlighting, prompt score.
        - **Fine-grained state tracking:** container open/closed states, temporal event logging, state-change overlay flags.
        - **Sports player/ball tracking:** ball trajectory, nearest-player possession candidate events.
        - **Wildlife observation:** animal class/species attributes, activity events in wildlife scenario.
        - **Retail shelf movement:** shelf-item in/out events, zone-oriented synthetic shelf background.
        - **Multi-camera handoff:** Camera A/B boundary, handoff hypothesis events with confidence.

        ### Evaluation
        - Approximate HOTA, IDF1, MOTA, ID switches, FP/FN, recovery rate, latency, and FPS are computed against synthetic ground truth.
        - For production, replace the simulated detector with a licensed YOLO/open-vocabulary detector and use official MOTChallenge-style evaluators.
        """
    )

    st.info(
        "This app intentionally avoids external API keys and heavyweight model downloads so it runs immediately. "
        "The synthetic engine is designed as a working orchestration/demo layer that can later be swapped with "
        "real detector weights while preserving the tracker, dashboard, event log, and export modules."
    )