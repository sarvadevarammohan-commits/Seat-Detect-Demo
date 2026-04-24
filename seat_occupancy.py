"""
Seat Occupancy Detection — Portable, Offline, Accurate
=======================================================
Uses YOLOv8-nano locally (no cloud API, no internet after first model download).
Works with a static camera. Detects persons and checks overlap with user-defined
seat zones using IoU. Temporal smoothing prevents flicker.

Supports THREE overlap-resolution modes for elevated/CCTV cameras:
  - rectangle (default): Original rectangle-overlap method (best for eye-level cameras)
  - anchor:    Bottom-center anchor point of person must fall inside seat zone
               (best for CCTV / elevated cameras with seats in a row)
  - polygon:   Define 4-point polygon (trapezoid) per seat that follows perspective
               (most accurate for permanent CCTV installs)
  - exclusive: Rectangle overlap but each person is assigned to ONE best-matching seat

Usage:
  1) pip install -r requirements.txt
  2) python seat_occupancy.py --setup --mode anchor       # draw seat zones
  3) python seat_occupancy.py --mode anchor               # run live detection

The YOLOv8-nano model (~6 MB) is auto-downloaded on first run and cached locally.
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# All paths are relative to THIS script's directory → fully portable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "seats_config.json"
LAYOUT_PATH = SCRIPT_DIR / "layout_status.png"
MODEL_NAME = "yolov8n.pt"  # auto-downloaded + cached by ultralytics

# COCO class-id for 'person'
PERSON_CLASS_ID = 0

# ---------------------------------------------------------------------------
# Detection confidence & overlap thresholds
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.45       # minimum YOLO confidence to accept a detection
IOU_THRESHOLD = 0.05        # Lowered to 5% for better sensitivity 
SMOOTHING_WINDOW = 7        # number of frames for temporal smoothing
OCCUPIED_VOTE_RATIO = 0.4   # ratio of "occupied" votes needed to declare occupied

# Valid detection modes
VALID_MODES = ("rectangle", "anchor", "polygon", "exclusive")


# ========================== Helper Functions ================================

def open_camera(source: int = 0) -> cv2.VideoCapture:
    """Open webcam with fallback across indices and backends."""
    indices = [source, 0, 1, 2]
    seen = set()
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(0)  # auto

    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            # warm-up
            for _ in range(5):
                cap.read()
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[INFO] Camera opened: index={idx}")
                return cap
            cap.release()
    raise SystemExit("ERROR: Could not open webcam. Close other apps using the camera and retry.")


# ========================== Overlap / Matching ==============================

def compute_iou(boxA: Tuple[int, ...], boxB: Tuple[int, ...]) -> float:
    """Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / min(areaA, areaB)  # use min-area ratio for better sensitivity


def overlap_ratio(person_box: Tuple[int, ...], seat_box: Tuple[int, ...]) -> float:
    """What fraction of the seat zone is covered by the person detection."""
    xa = max(person_box[0], seat_box[0])
    ya = max(person_box[1], seat_box[1])
    xb = min(person_box[2], seat_box[2])
    yb = min(person_box[3], seat_box[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    seat_area = max(1, (seat_box[2] - seat_box[0]) * (seat_box[3] - seat_box[1]))
    return inter / seat_area


def center_in_box(person_box: Tuple[int, ...], seat_box: Tuple[int, ...]) -> bool:
    """Check if the center of the person detection is inside the seat zone."""
    cx = (person_box[0] + person_box[2]) // 2
    cy = (person_box[1] + person_box[3]) // 2
    return seat_box[0] <= cx <= seat_box[2] and seat_box[1] <= cy <= seat_box[3]


def person_in_seat(person_box: Tuple[int, ...], seat_box: Tuple[int, ...]) -> bool:
    """
    Robust check: person is 'in' a seat if:
      - IoU >= threshold, OR
      - overlap of seat area >= 20%, OR
      - center of person bbox is inside seat zone
    """
    iou = compute_iou(person_box, seat_box)
    olap = overlap_ratio(person_box, seat_box)
    ctr = center_in_box(person_box, seat_box)
    return iou >= IOU_THRESHOLD or olap >= 0.15 or ctr


# ---------------------------------------------------------------------------
# ANCHOR MODE: use bottom-center of person bounding box
# ---------------------------------------------------------------------------
def bottom_center_of(person_box: Tuple[int, ...]) -> Tuple[int, int]:
    """Return (cx, bottom_y) — the bottom-center of the person bounding box.
    This approximates a person's 'ground contact point' and stays within the
    seat they're actually sitting in, even when viewed from an elevated angle.
    """
    cx = (person_box[0] + person_box[2]) // 2
    by = person_box[3]  # bottom edge y
    return (cx, by)


def anchor_in_rect(person_box: Tuple[int, ...], seat_box: Tuple[int, ...]) -> bool:
    """True if bottom-center of person box falls inside the seat rectangle."""
    cx, by = bottom_center_of(person_box)
    return seat_box[0] <= cx <= seat_box[2] and seat_box[1] <= by <= seat_box[3]


# ---------------------------------------------------------------------------
# POLYGON MODE: point-in-polygon test using cv2
# ---------------------------------------------------------------------------
def anchor_in_polygon(person_box: Tuple[int, ...], polygon_pts: np.ndarray) -> bool:
    """True if bottom-center of person box is inside the polygon."""
    cx, by = bottom_center_of(person_box)
    # cv2.pointPolygonTest returns > 0 if inside, 0 if on edge, < 0 if outside
    dist = cv2.pointPolygonTest(polygon_pts, (float(cx), float(by)), False)
    return dist >= 0


# ---------------------------------------------------------------------------
# EXCLUSIVE MODE: one person → best-matching seat only
# ---------------------------------------------------------------------------
def exclusive_assignment(
    detections: List[Tuple[int, ...]],
    seat_boxes: List[Tuple[int, ...]],
) -> List[bool]:
    """Assign each person to at most ONE seat (the best IoU match).
    Returns a list of booleans per seat.
    """
    n_seats = len(seat_boxes)
    occupied = [False] * n_seats

    for det in detections:
        pbox = det[:4]
        best_idx = -1
        best_score = 0.0
        for i, sbox in enumerate(seat_boxes):
            score = overlap_ratio(pbox, sbox)
            iou = compute_iou(pbox, sbox)
            # combined score
            combined = max(score, iou)
            if combined > best_score and combined >= IOU_THRESHOLD:
                best_score = combined
                best_idx = i
        if best_idx >= 0:
            occupied[best_idx] = True

    return occupied


# ========================== Seat Configuration ==============================

def draw_seat_zones_rect(frame: np.ndarray, num_seats: int) -> List[Dict]:
    """Interactive UI to draw exactly `num_seats` seat zones as rectangles."""
    boxes = []
    drawing = {"start": None, "curr": None}
    canvas_base = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["start"] = (x, y)
            drawing["curr"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["start"] is not None:
            drawing["curr"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["start"] is not None:
            sx, sy = drawing["start"]
            ex, ey = x, y
            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)
            if x2 - x1 > 10 and y2 - y1 > 10:
                boxes.append((x1, y1, x2, y2))
            drawing["start"] = None
            drawing["curr"] = None

    win = f"Draw {num_seats} Seat Zone(s) | Enter=Save | C=Clear | U=Undo | Esc=Quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = canvas_base.copy()
        # Instructions
        cv2.putText(canvas, f"Draw exactly {num_seats} seat zone(s). Current: {len(boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Enter=Save | C=Clear | U=Undo last | Esc=Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw existing boxes
        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(canvas, f"S{i}", (x1 + 4, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        # Draw in-progress rectangle
        if drawing["start"] and drawing["curr"]:
            cv2.rectangle(canvas, drawing["start"], drawing["curr"], (0, 255, 255), 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if len(boxes) == num_seats:
                break
            else:
                print(f"[WARN] Draw exactly {num_seats} seat zone(s). Currently have {len(boxes)}.")
        elif key == ord("c"):
            boxes.clear()
        elif key == ord("u") and boxes:
            boxes.pop()
        elif key == 27:
            raise SystemExit("Setup cancelled.")

    cv2.destroyWindow(win)

    seats = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        seats.append({"id": f"S{i}", "box": [x1, y1, x2, y2]})
    return seats


def draw_seat_zones_polygon(frame: np.ndarray, num_seats: int) -> List[Dict]:
    """Interactive UI to draw `num_seats` 4-point polygon seat zones.
    
    Click 4 corners per seat (in order: top-left, top-right, bottom-right, bottom-left
    or any clockwise/counter-clockwise order). Each group of 4 clicks = one seat.
    """
    all_polys = []  # list of list of (x,y)
    current_pts = []
    canvas_base = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append((x, y))
            if len(current_pts) == 4:
                all_polys.append(list(current_pts))
                current_pts.clear()

    win = (f"Polygon Mode: Click 4 corners per seat ({num_seats} seats) | "
           f"Enter=Save | C=Clear | U=Undo seat | Esc=Quit")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = canvas_base.copy()
        # Instructions
        completed = len(all_polys)
        in_prog = len(current_pts)
        cv2.putText(canvas,
                    f"Seats done: {completed}/{num_seats} | Current clicks: {in_prog}/4",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    "Click 4 corners per seat (any order) | Enter=Save | C=Clear | U=Undo | Esc=Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw completed polygons
        colors = [
            (255, 100, 0), (0, 200, 100), (100, 100, 255),
            (255, 200, 0), (200, 0, 200), (0, 200, 200),
            (150, 255, 50), (255, 50, 150), (50, 150, 255), (200, 200, 50),
        ]
        for i, poly in enumerate(all_polys):
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            col = colors[i % len(colors)]
            cv2.polylines(canvas, [pts], isClosed=True, color=col, thickness=2)
            # Seat label at centroid
            cx = sum(p[0] for p in poly) // 4
            cy = sum(p[1] for p in poly) // 4
            cv2.putText(canvas, f"S{i+1}", (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        # Draw in-progress points
        for pt in current_pts:
            cv2.circle(canvas, pt, 5, (0, 255, 255), -1)
        if len(current_pts) >= 2:
            for j in range(len(current_pts) - 1):
                cv2.line(canvas, current_pts[j], current_pts[j+1], (0, 255, 255), 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if completed == num_seats:
                break
            else:
                print(f"[WARN] Need exactly {num_seats} seat polygon(s). "
                      f"Currently have {completed}.")
        elif key == ord("c"):
            all_polys.clear()
            current_pts.clear()
        elif key == ord("u"):
            if current_pts:
                current_pts.pop()
            elif all_polys:
                all_polys.pop()
        elif key == 27:
            raise SystemExit("Setup cancelled.")

    cv2.destroyWindow(win)

    seats = []
    for i, poly in enumerate(all_polys, start=1):
        seats.append({"id": f"S{i}", "polygon": poly})
    return seats


def save_config(seats: List[Dict], mode: str) -> None:
    data = {"mode": mode, "seats": seats}
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {len(seats)} seat zone(s) (mode={mode}) to {CONFIG_PATH}")


def load_config() -> Tuple[List[Dict], str]:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"ERROR: {CONFIG_PATH} not found.\n"
            f"Run:  python seat_occupancy.py --setup   to create seat zones first."
        )
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seats = data.get("seats", [])
    mode = data.get("mode", "rectangle")
    if not seats:
        raise SystemExit("ERROR: No seats defined in config. Run --setup again.")
    print(f"[INFO] Loaded {len(seats)} seat zone(s) (mode={mode}) from {CONFIG_PATH}")
    return seats, mode


# ========================== Layout Renderer =================================

def render_layout(seats: List[Dict], occupied: List[bool]) -> np.ndarray:
    """Generate a clean dashboard image showing seat occupancy status."""
    n = len(seats)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 160, 120
    pad = 20
    header_h = 60
    w = pad + cols * (cell_w + pad)
    h = header_h + pad + rows * (cell_h + pad) + 30
    img = np.full((h, w, 3), 30, dtype=np.uint8)  # dark background

    # Header
    cv2.putText(img, "SEAT OCCUPANCY MONITOR", (pad, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)

    for i, seat in enumerate(seats):
        r, c = divmod(i, cols)
        x1 = pad + c * (cell_w + pad)
        y1 = header_h + pad + r * (cell_h + pad)
        x2 = x1 + cell_w
        y2 = y1 + cell_h
        occ = occupied[i]

        # Seat block
        fill_color = (50, 50, 200) if occ else (50, 180, 50)
        cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)

        # Label
        label = seat["id"]
        status = "OCCUPIED" if occ else "FREE"
        cv2.putText(img, label, (x1 + 10, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, status, (x1 + 10, y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Summary
    occ_count = sum(occupied)
    free_count = n - occ_count
    summary = f"Occupied: {occ_count}  |  Free: {free_count}  |  Total: {n}"
    cv2.putText(img, summary, (pad, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return img


# ========================== Main Loop =======================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Seat Occupancy Detection — Portable & Offline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--setup", action="store_true",
                    help="Draw seat zones interactively on the first camera frame.")
    ap.add_argument("--camera", type=int, default=0,
                    help="Camera index (default: 0).")
    ap.add_argument("--confidence", type=float, default=CONF_THRESHOLD,
                    help=f"YOLO confidence threshold (default: {CONF_THRESHOLD}).")
    ap.add_argument("--show-fps", action="store_true",
                    help="Show FPS on the live view.")
    ap.add_argument("--mode", type=str, default=None,
                    choices=VALID_MODES,
                    help=("Detection mode for seat matching:\n"
                          "  rectangle  - Original IoU/overlap method (eye-level cameras)\n"
                          "  anchor     - Bottom-center point match (best for elevated/CCTV)\n"
                          "  polygon    - 4-point polygon zones (most accurate for CCTV)\n"
                          "  exclusive  - Rectangle overlap, one person = one seat only\n"
                          "Default: rectangle. For CCTV at 45°, use 'anchor' or 'polygon'."))
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load YOLO model (local, cached automatically by ultralytics)
    # ------------------------------------------------------------------
    print("[INFO] Loading YOLOv8-nano model (first run downloads ~6 MB)...")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ERROR: ultralytics not installed.\n"
            "Run:  pip install -r requirements.txt"
        )
    model = YOLO(MODEL_NAME)
    print("[INFO] Model loaded.")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    cap = open_camera(args.camera)
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise SystemExit("ERROR: Could not read first frame from camera.")

    # ------------------------------------------------------------------
    # Setup or load seat config
    # ------------------------------------------------------------------
    if args.setup:
        mode = args.mode or "rectangle"

        # Ask user how many seats to define
        while True:
            try:
                num_seats = int(input("\nHow many seats do you want to define? "))
                if num_seats < 1:
                    print("[WARN] Enter at least 1.")
                    continue
                break
            except ValueError:
                print("[WARN] Please enter a valid number.")

        if mode == "polygon":
            print(f"[INFO] POLYGON MODE: Click 4 corners per seat ({num_seats} seats).")
            print("[INFO] Click corners in order (clockwise or counter-clockwise).")
            seat_defs = draw_seat_zones_polygon(first_frame, num_seats)
        else:
            print(f"[INFO] Draw exactly {num_seats} seat zone(s) on the camera frame.")
            seat_defs = draw_seat_zones_rect(first_frame, num_seats)

        save_config(seat_defs, mode)
    else:
        seat_defs, saved_mode = load_config()
        mode = args.mode or saved_mode

    n_seats = len(seat_defs)

    # Prepare seat geometry depending on mode
    seat_boxes = []       # for rectangle/anchor/exclusive modes
    seat_polygons = []    # for polygon mode

    if mode == "polygon":
        for s in seat_defs:
            if "polygon" in s:
                pts = np.array(s["polygon"], np.int32)
                seat_polygons.append(pts)
            elif "box" in s:
                # Fallback: convert box to polygon
                x1, y1, x2, y2 = s["box"]
                pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.int32)
                seat_polygons.append(pts)
    else:
        for s in seat_defs:
            if "box" in s:
                seat_boxes.append(tuple(s["box"]))
            elif "polygon" in s:
                # Fallback: compute bounding rect of polygon
                pts = np.array(s["polygon"], np.int32)
                x, y, w_r, h_r = cv2.boundingRect(pts)
                seat_boxes.append((x, y, x + w_r, y + h_r))

    # Temporal smoothing: per-seat history of raw occupied booleans
    history: List[deque] = [deque(maxlen=SMOOTHING_WINDOW) for _ in range(n_seats)]
    smoothed_occupied = [False] * n_seats

    # ------------------------------------------------------------------
    # Windows
    # ------------------------------------------------------------------
    cv2.namedWindow("Seat Occupancy — Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seat Occupancy — Live", 960, 540)
    cv2.namedWindow("Seat Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seat Dashboard", 900, 300)

    fps_timer = time.time()
    fps_count = 0
    fps_display = 0.0

    mode_labels = {
        "rectangle": "RECT",
        "anchor": "ANCHOR (bottom-center)",
        "polygon": "POLYGON",
        "exclusive": "EXCLUSIVE",
    }
    print(f"[INFO] Detection mode: {mode} — {mode_labels.get(mode, mode)}")
    print("[INFO] Running... Press Q or Esc to quit, S to save layout image.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # ==============================================================
        # Run YOLO person detection on EVERY frame (local model is fast)
        # ==============================================================
        results = model(frame, conf=args.confidence, classes=[PERSON_CLASS_ID],
                        verbose=False)
        detections = []  # list of (x1, y1, x2, y2, confidence)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                detections.append((int(x1), int(y1), int(x2), int(y2), conf))

        # ==============================================================
        # Check seat occupancy based on selected MODE
        # ==============================================================
        if mode == "anchor":
            # ANCHOR MODE: bottom-center of person must be inside seat rect
            raw_occupied = [False] * n_seats
            for i, sbox in enumerate(seat_boxes):
                for det in detections:
                    pbox = det[:4]
                    if anchor_in_rect(pbox, sbox) or overlap_ratio(pbox, sbox) >= 0.15:
                        raw_occupied[i] = True
                        break

        elif mode == "polygon":
            # POLYGON MODE: bottom-center of person must be inside polygon
            raw_occupied = [False] * n_seats
            for i, poly_pts in enumerate(seat_polygons):
                for det in detections:
                    pbox = det[:4]
                    if anchor_in_polygon(pbox, poly_pts) or center_in_box(pbox, cv2.boundingRect(poly_pts)):
                        raw_occupied[i] = True
                        break

        elif mode == "exclusive":
            # EXCLUSIVE MODE: each person → best matching seat only
            raw_occupied = exclusive_assignment(detections, seat_boxes)

        else:
            # RECTANGLE MODE (default/original)
            raw_occupied = [False] * n_seats
            for i, sbox in enumerate(seat_boxes):
                for det in detections:
                    pbox = det[:4]
                    if person_in_seat(pbox, sbox):
                        raw_occupied[i] = True
                        break

        # Temporal smoothing
        for i in range(n_seats):
            history[i].append(raw_occupied[i])
            votes = sum(history[i])
            smoothed_occupied[i] = votes >= len(history[i]) * OCCUPIED_VOTE_RATIO

        # ==============================================================
        # Visualize live feed
        # ==============================================================
        vis = frame.copy()

        # Mode indicator on frame
        cv2.putText(vis, f"Mode: {mode.upper()}", (10, vis.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw seat zones depending on mode
        if mode == "polygon":
            for i, poly_pts in enumerate(seat_polygons):
                occ = smoothed_occupied[i]
                color = (0, 0, 220) if occ else (0, 200, 0)
                pts = poly_pts.reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=3)
                # Semi-transparent fill
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
                # Label at centroid
                cx = int(poly_pts[:, 0].mean())
                cy = int(poly_pts[:, 1].mean())
                label = f"{seat_defs[i]['id']}: {'OCC' if occ else 'FREE'}"
                cv2.putText(vis, label, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            for i, sbox in enumerate(seat_boxes):
                x1, y1, x2, y2 = sbox
                occ = smoothed_occupied[i]
                color = (0, 0, 220) if occ else (0, 200, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                label = f"{seat_defs[i]['id']}: {'OCCUPIED' if occ else 'FREE'}"
                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis, (x1, max(0, y1 - th - 10)),
                              (x1 + tw + 6, max(0, y1 - 2)), color, -1)
                cv2.putText(vis, label, (x1 + 3, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw person detections — premium compact style
        for x1, y1, x2, y2, conf in detections:
            w = x2 - x1
            h = y2 - y1
            
            # Cap drawing aspect ratio to prevent 'long' boxes
            draw_h = min(h, int(w * 2.2))
            draw_y1 = y1 + (h - draw_h) // 2
            
            # Draw compact 'bracket' style box
            bw = int(w * 0.70)
            bh = int(draw_h * 0.70)
            bx = x1 + (w - bw) // 2
            by = draw_y1 + (draw_h - bh) // 2
            
            color = (0, 204, 255)  # yellow/gold in BGR: (0, 255, 255) or (0, 204, 255)
            thickness = 2
            length = max(5, int(min(bw, bh) * 0.25))
            
            # Draw corner brackets
            # Top-left
            cv2.line(vis, (bx, by), (bx + length, by), color, thickness)
            cv2.line(vis, (bx, by), (bx, by + length), color, thickness)
            # Top-right
            cv2.line(vis, (bx + bw - length, by), (bx + bw, by), color, thickness)
            cv2.line(vis, (bx + bw, by), (bx + bw, by + length), color, thickness)
            # Bottom-right
            cv2.line(vis, (bx + bw, by + bh - length), (bx + bw, by + bh), color, thickness)
            cv2.line(vis, (bx + bw - length, by + bh), (bx + bw, by + bh), color, thickness)
            # Bottom-left
            cv2.line(vis, (bx, by + bh - length), (bx, by + bh), color, thickness)
            cv2.line(vis, (bx + length, by + bh), (bx, by + bh), color, thickness)

            # Label (conf)
            cv2.putText(vis, f"{conf*100:.0f}%", (bx, max(15, by - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
            # Show anchor point for anchor/polygon modes
            if mode in ("anchor", "polygon"):
                cx, by_p = bottom_center_of((x1, y1, x2, y2))
                cv2.circle(vis, (cx, by_p), 5, (0, 255, 255), -1)
                cv2.circle(vis, (cx, by_p), 6, (0, 0, 0), 1)

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()
        if args.show_fps:
            cv2.putText(vis, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Dashboard
        layout_img = render_layout(seat_defs, smoothed_occupied)

        cv2.imshow("Seat Occupancy — Live", vis)
        cv2.imshow("Seat Dashboard", layout_img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            cv2.imwrite(str(LAYOUT_PATH), layout_img)
            print(f"[INFO] Saved layout image: {LAYOUT_PATH}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
