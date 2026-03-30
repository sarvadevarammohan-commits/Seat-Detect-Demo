# Seat Occupancy Monitor — Real-Time AI Detection (Web)

Real-time seat occupancy detection using **YOLOv8-nano** running entirely **in your browser** via ONNX Runtime Web.  
No server, no cloud, no API keys — just your camera and AI.

## ✨ Features

- **100% Client-Side AI** — YOLOv8-nano (~12 MB ONNX) runs in-browser via WebAssembly
- **Camera API** — Uses browser `getUserMedia` for live webcam access
- **Canvas 2D Overlay** — Real-time bounding box + seat zone rendering
- **4 Detection Modes** — Rectangle, Anchor, Polygon, Exclusive (same as the original Python version)
- **Temporal Smoothing** — Prevents flickering occupancy changes
- **Interactive Setup** — Draw seat zones directly on the camera feed
- **Deploy to Vercel** — Single `vercel deploy` and you're live
- **Premium Dark UI** — Glassmorphism, gradient mesh, smooth animations

---

## 🚀 Quick Start

### Prerequisites
- **Node.js 18+** installed ([download](https://nodejs.org/))
- A webcam / USB camera

### 1. Install dependencies
```bash
npm install
```

### 2. Run locally
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000)

### 3. Use the app
1. **Configure** — Select camera, choose detection mode, set number of seats
2. **Setup** — Draw seat zones on the live camera feed (drag rectangles or click 4 polygon corners)
3. **Detect** — AI model loads in browser and runs real-time person detection

---

## 🌐 Deploy to Vercel

```bash
npx vercel deploy
```

Or push to GitHub and connect to [Vercel](https://vercel.com) for automatic deployments.

> **Note:** The ONNX model (~12 MB) is served from the `public/` folder as a static asset.

---

## 🎯 Detection Modes

| Mode | Best For | How It Works |
|------|----------|-------------|
| **Rectangle** | Eye-level cameras | IoU/overlap/center-check |
| **Anchor** ⭐ | CCTV / elevated cameras | Bottom-center point of person bbox falls inside seat zone |
| **Polygon** | Permanent CCTV installs | 4-point trapezoid zones following perspective |
| **Exclusive** | Moderate overlap | Rectangle overlap, one person = one seat only |

---

## 🏗️ Architecture

```
Browser (Client-Side Only)
├── Camera API (getUserMedia)
├── Canvas 2D API (rendering overlays)
├── ONNX Runtime Web (WASM backend)
│   └── YOLOv8-nano model (person detection)
├── Occupancy Logic (4 modes + temporal smoothing)
└── React UI (Next.js)
```

**No server-side processing** — everything runs in the browser:
- Camera frames are captured via Canvas
- YOLO inference runs on ONNX Runtime Web (WebAssembly)
- Occupancy is computed client-side with the same algorithms as the Python version

---

## 📁 Project Structure

| Path | Purpose |
|------|---------|
| `src/app/page.tsx` | Main app (configure → setup → detect phases) |
| `src/app/globals.css` | Design system (dark theme, glassmorphism) |
| `src/app/layout.tsx` | Root layout with SEO metadata |
| `src/lib/yolo.ts` | ONNX model loading + YOLOv8 inference |
| `src/lib/occupancy.ts` | Seat matching logic (4 modes) + smoothing |
| `src/lib/types.ts` | Shared TypeScript types |
| `public/yolov8n.onnx` | YOLOv8-nano ONNX model |
| `next.config.mjs` | Next.js + WASM configuration |

### Legacy Python Files (kept for reference)
| File | Purpose |
|------|---------|
| `seat_occupancy.py` | Original Python detection script |
| `two_seat_occupancy.py` | 2-seat Roboflow API demo |
| `*.bat` | Windows batch scripts (replaced by web UI) |

---

## 🔧 Configuration (via Web UI)

All settings that were previously CLI arguments or batch file prompts are now interactive web controls:

| Original (Python/Batch) | Web Equivalent |
|------------------------|----------------|
| `--mode rectangle\|anchor\|polygon\|exclusive` | Mode selection cards |
| `--camera N` | Camera dropdown selector |
| `--confidence F` | Confidence threshold slider |
| `--show-fps` | FPS toggle switch |
| `--setup` (draw seat zones) | Interactive canvas drawing |
| `"How many seats?"` prompt | Number input field |
| `setup.bat` / `run_setup.bat` | "Start Camera & Draw Seat Zones" button |
| `run_detect.bat` | "Save & Start Detection" button |

---

## 📋 Requirements

- Node.js 18+
- Modern browser (Chrome, Edge, Firefox, Safari 16.4+)
- Webcam
- No internet needed after initial load (ONNX model cached by browser)
