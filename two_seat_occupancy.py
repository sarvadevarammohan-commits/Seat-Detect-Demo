import argparse
import base64
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import requests


WORKSPACE = "ramus-workspace"
WORKFLOW_ID = "detect-count-and-visualize"
SERVERLESS_URL = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_ID}"
API_KEY_ENV = "ROBOFLOW_API_KEY"
CONFIG_PATH = Path("seat_occupancy_demo") / "two_seats.json"
OUT_LAYOUT = Path("seat_occupancy_demo") / "sample_layout_status.png"


def iter_dicts(v: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(v, dict):
        yield v
        for vv in v.values():
            yield from iter_dicts(vv)
    elif isinstance(v, list):
        for item in v:
            yield from iter_dicts(item)


def open_camera() -> cv2.VideoCapture:
    indices = [0, 1, 2]
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(0)

    for idx in indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(10):
                cap.read()
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
    raise SystemExit("Could not open webcam. Close Camera/Teams/Zoom and retry.")


def call_person_detector(frame_bgr: np.ndarray, api_key: str) -> List[Tuple[int, int, int, int, float]]:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return []
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    payload = {
        "api_key": api_key,
        "inputs": {"image": {"type": "base64", "value": img_b64}},
        "use_cache": True,
    }
    resp = requests.post(SERVERLESS_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    dets = []
    for d in iter_dicts(data):
        if d.get("class") == "person":
            try:
                x = float(d["x"])
                y = float(d["y"])
                w = float(d["width"])
                h = float(d["height"])
                conf = float(d.get("confidence", 1.0))
            except (KeyError, ValueError, TypeError):
                continue
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            dets.append((x1, y1, x2, y2, conf))
    return dets


def pick_two_seats(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
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

    win = "Draw 2 seats: drag rectangles, Enter=save, C=clear, Esc=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = canvas_base.copy()
        cv2.putText(
            canvas,
            f"Draw exactly 2 seat boxes. Current: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(canvas, f"S{i}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if drawing["start"] is not None and drawing["curr"] is not None:
            sx, sy = drawing["start"]
            cx, cy = drawing["curr"]
            cv2.rectangle(canvas, (sx, sy), (cx, cy), (0, 255, 255), 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # Enter
            if len(boxes) == 2:
                break
        elif key == ord("c"):
            boxes = []
        elif key == 27:
            raise SystemExit("Cancelled.")

    cv2.destroyWindow(win)
    return boxes


def save_two_seat_config(boxes: List[Tuple[int, int, int, int]]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seats": [{"id": "S1", "box": list(boxes[0])}, {"id": "S2", "box": list(boxes[1])}]}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_two_seat_config() -> List[Tuple[int, int, int, int]]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing {CONFIG_PATH}. Run with --setup first.")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [tuple(s["box"]) for s in cfg["seats"]]


def point_in_box(pt: Tuple[int, int], box: Tuple[int, int, int, int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def render_layout(occupied: List[bool]) -> np.ndarray:
    # Simple generated layout: 2 seat blocks + status labels.
    img = np.full((320, 700, 3), 245, dtype=np.uint8)
    cv2.putText(img, "Seat Occupancy Layout", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)

    boxes = [(80, 110, 300, 250), (390, 110, 610, 250)]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        occ = occupied[i]
        color = (0, 0, 255) if occ else (0, 180, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 2)
        label = f"S{i+1}: OCCUPIED" if occ else f"S{i+1}: FREE"
        cv2.putText(img, label, (x1 + 20, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    OUT_LAYOUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_LAYOUT), img)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-seat occupancy demo (person-only).")
    ap.add_argument("--setup", action="store_true", help="Draw 2 seat boxes on first frame and save config.")
    ap.add_argument("--detect-every", type=int, default=10, help="Run detection every N frames.")
    args = ap.parse_args()

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Set {API_KEY_ENV} environment variable first.")

    cap = open_camera()
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise SystemExit("Could not read first webcam frame.")

    if args.setup:
        seat_boxes = pick_two_seats(first)
        save_two_seat_config(seat_boxes)
        print(f"Saved seat config to {CONFIG_PATH}")
    else:
        seat_boxes = load_two_seat_config()

    frame_idx = 0
    last_error = ""
    dets: List[Tuple[int, int, int, int, float]] = []
    occupied = [False, False]
    busy = False
    lock = threading.Lock()
    pending = None

    cv2.namedWindow("Live Person Detect (2-seat demo)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Layout Status", cv2.WINDOW_NORMAL)

    def detect_worker(frame_local):
        nonlocal pending, last_error, busy
        try:
            res = call_person_detector(frame_local, api_key=api_key)
            with lock:
                pending = res
            last_error = ""
        except Exception as e:
            last_error = str(e)
        finally:
            busy = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1

        if frame_idx % args.detect_every == 0 and not busy:
            busy = True
            threading.Thread(target=detect_worker, args=(frame.copy(),), daemon=True).start()

        with lock:
            if pending is not None:
                dets = pending
                pending = None

        # seat occupancy from bottom-center of each person box
        occupied = [False, False]
        for x1, y1, x2, y2, conf in dets:
            px = int((x1 + x2) / 2)
            py = int(y2)
            if point_in_box((px, py), seat_boxes[0]):
                occupied[0] = True
            if point_in_box((px, py), seat_boxes[1]):
                occupied[1] = True

        vis = frame.copy()
        # draw seat boxes
        for i, box in enumerate(seat_boxes, start=1):
            x1, y1, x2, y2 = box
            color = (0, 0, 255) if occupied[i - 1] else (0, 180, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            txt = f"S{i}: OCCUPIED" if occupied[i - 1] else f"S{i}: FREE"
            cv2.putText(vis, txt, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw person detections
        for x1, y1, x2, y2, conf in dets:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                vis,
                f"person {conf*100:.0f}%",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        if last_error:
            cv2.putText(vis, f"API error: {last_error[:65]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        layout_img = render_layout(occupied)
        cv2.imshow("Live Person Detect (2-seat demo)", vis)
        cv2.imshow("Layout Status", layout_img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            cv2.imwrite(str(OUT_LAYOUT), layout_img)
            print(f"Saved layout image: {OUT_LAYOUT}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

