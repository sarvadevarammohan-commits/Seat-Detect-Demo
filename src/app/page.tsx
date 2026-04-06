'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import type { AppPhase, DetectionMode, SeatConfig, Detection } from '@/lib/types';
import { MODE_INFO } from '@/lib/types';
import { YOLODetector } from '@/lib/yolo';
import { checkOccupancy, TemporalSmoother } from '@/lib/occupancy';

/* ================================================================
   MAIN PAGE — state machine: configure → setup → detect
   Single persistent canvas & video to avoid ref lifecycle issues.
   ================================================================ */
export default function Home() {
  const [phase, setPhase] = useState<AppPhase>('configure');
  const [mode, setMode] = useState<DetectionMode>('anchor');
  const [numSeats, setNumSeats] = useState(2);
  const [seats, setSeats] = useState<SeatConfig[]>([]);
  const [confidence, setConfidence] = useState(0.4);
  const [targetFps, setTargetFps] = useState(15);
  const [showFps, setShowFps] = useState(true);
  const [cameraId, setCameraId] = useState('');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState('');

  // Model state
  const [modelStatus, setModelStatus] = useState<
    'idle' | 'loading' | 'ready' | 'error'
  >('idle');
  const [modelProgress, setModelProgress] = useState(0);
  const detectorRef = useRef<YOLODetector | null>(null);

  // Detection state
  const [detections, setDetections] = useState<Detection[]>([]);
  const [occupied, setOccupied] = useState<boolean[]>([]);
  const [fps, setFps] = useState(0);

  // PERSISTENT refs — video & canvas never get destroyed
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runningRef = useRef(false);
  const smootherRef = useRef<TemporalSmoother | null>(null);
  const phaseRef = useRef(phase);
  useEffect(() => { phaseRef.current = phase; }, [phase]);

  /* ─── enumerate cameras ──────────────────── */
  const enumerateCameras = useCallback(async () => {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const vids = devs.filter((d) => d.kind === 'videoinput');
      setCameras(vids);
      if (vids.length > 0 && !cameraId) setCameraId(vids[0].deviceId);
      return vids;
    } catch {
      return [];
    }
  }, [cameraId]);

  useEffect(() => { enumerateCameras(); }, [enumerateCameras]);

  /* ─── start camera ───────────────────────── */
  const startCamera = useCallback(async () => {
    setCameraError('');
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    try {
      const constraints: MediaStreamConstraints = {
        video: cameraId
          ? { deviceId: { exact: cameraId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('Camera timed out')), 8000);
          const onPlaying = () => {
            clearTimeout(timeout);
            video.removeEventListener('playing', onPlaying);
            resolve();
          };
          video.addEventListener('playing', onPlaying);
          video.play().catch(reject);
        });
        setCameraReady(true);
      }
      // Re-enumerate to get labels after permission
      const devs = await navigator.mediaDevices.enumerateDevices();
      const vids = devs.filter((d) => d.kind === 'videoinput');
      setCameras(vids);
      if (vids.length > 0 && !cameraId) setCameraId(vids[0].deviceId);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      console.error('Camera error:', err);
      if (msg.includes('Permission') || msg.includes('NotAllowed')) {
        setCameraError('Camera permission denied. Please allow camera access in your browser settings and reload.');
      } else if (msg.includes('NotFound') || msg.includes('DevicesNotFound')) {
        setCameraError('No camera found. Please connect a camera and try again.');
      } else if (msg.includes('NotReadable') || msg.includes('TrackStartError')) {
        setCameraError('Camera is in use by another application. Close other apps using the camera and try again.');
      } else {
        setCameraError(`Camera error: ${msg}. Try a different camera or reload.`);
      }
      setCameraReady(false);
      throw err;
    }
  }, [cameraId]);

  const stopCamera = useCallback(() => {
    runningRef.current = false;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraReady(false);
  }, []);

  /* ─── load model ─────────────────────────── */
  const loadModel = useCallback(async () => {
    if (detectorRef.current?.isReady) { setModelStatus('ready'); return; }
    setModelStatus('loading');
    setModelProgress(0);
    try {
      const det = new YOLODetector();
      await det.load('/yolov8n.onnx', (p) => setModelProgress(p));
      detectorRef.current = det;
      setModelStatus('ready');
    } catch (err) {
      console.error('Model load error:', err);
      setModelStatus('error');
    }
  }, []);

  /* ─── SETUP phase state ──────────────────── */
  const [setupBoxes, setSetupBoxes] = useState<[number, number, number, number][]>([]);
  const [setupPolygons, setSetupPolygons] = useState<[number, number][][]>([]);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);
  const [polyPoints, setPolyPoints] = useState<[number, number][]>([]);

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phaseRef.current !== 'setup') return;
    if (mode === 'polygon') {
      const { x, y } = getCanvasCoords(e);
      const newPts = [...polyPoints, [x, y] as [number, number]];
      if (newPts.length === 4) {
        if (setupPolygons.length < numSeats) setSetupPolygons((prev) => [...prev, newPts]);
        setPolyPoints([]);
      } else {
        setPolyPoints(newPts);
      }
    } else {
      if (setupBoxes.length >= numSeats) return;
      setDrawStart(getCanvasCoords(e));
      setDrawCurrent(getCanvasCoords(e));
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phaseRef.current !== 'setup') return;
    if (mode !== 'polygon' && drawStart) setDrawCurrent(getCanvasCoords(e));
  };

  const handleCanvasMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phaseRef.current !== 'setup') return;
    if (mode !== 'polygon' && drawStart) {
      const end = getCanvasCoords(e);
      const x1 = Math.min(drawStart.x, end.x), y1 = Math.min(drawStart.y, end.y);
      const x2 = Math.max(drawStart.x, end.x), y2 = Math.max(drawStart.y, end.y);
      if (x2 - x1 > 10 && y2 - y1 > 10 && setupBoxes.length < numSeats) {
        setSetupBoxes((prev) => [...prev, [x1, y1, x2, y2]]);
      }
      setDrawStart(null);
      setDrawCurrent(null);
    }
  };

  /* ─── UNIFIED render loop (setup + detect) ── */
  const rafRef = useRef<number>(0);
  const detectBusy = useRef(false);
  const fpsCountRef = useRef(0);
  const fpsTimeRef = useRef(performance.now());
  const lastInferenceRef = useRef(0);
  const seatsRef = useRef(seats);
  const modeRef = useRef(mode);
  const confRef = useRef(confidence);
  const targetFpsRef = useRef(targetFps);
  const detsRef = useRef(detections);
  const occRef = useRef(occupied);
  useEffect(() => { seatsRef.current = seats; }, [seats]);
  useEffect(() => { modeRef.current = mode; }, [mode]);
  useEffect(() => { confRef.current = confidence; }, [confidence]);
  useEffect(() => { targetFpsRef.current = targetFps; }, [targetFps]);
  useEffect(() => { detsRef.current = detections; }, [detections]);
  useEffect(() => { occRef.current = occupied; }, [occupied]);

  // Refs for setup drawing state (avoids stale closures)
  const setupBoxesRef = useRef(setupBoxes);
  const setupPolygonsRef = useRef(setupPolygons);
  const drawStartRef = useRef(drawStart);
  const drawCurrentRef = useRef(drawCurrent);
  const polyPointsRef = useRef(polyPoints);
  const numSeatsRef = useRef(numSeats);
  useEffect(() => { setupBoxesRef.current = setupBoxes; }, [setupBoxes]);
  useEffect(() => { setupPolygonsRef.current = setupPolygons; }, [setupPolygons]);
  useEffect(() => { drawStartRef.current = drawStart; }, [drawStart]);
  useEffect(() => { drawCurrentRef.current = drawCurrent; }, [drawCurrent]);
  useEffect(() => { polyPointsRef.current = polyPoints; }, [polyPoints]);
  useEffect(() => { numSeatsRef.current = numSeats; }, [numSeats]);

  useEffect(() => {
    if (phase !== 'setup' && phase !== 'detect') {
      cancelAnimationFrame(rafRef.current);
      return;
    }

    let active = true;
    runningRef.current = true;
    fpsCountRef.current = 0;
    fpsTimeRef.current = performance.now();

    const render = () => {
      if (!active) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) { rafRef.current = requestAnimationFrame(render); return; }

      const ctx = canvas.getContext('2d')!;
      const videoOk = video.readyState >= 2 && video.videoWidth > 0;

      if (videoOk) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      } else {
        if (canvas.width < 2) canvas.width = 960;
        if (canvas.height < 2) canvas.height = 540;
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#888';
        ctx.font = '18px Inter, sans-serif';
        ctx.fillText('Connecting to camera...', 20, 40);
        ctx.font = '14px Inter, sans-serif';
        ctx.fillStyle = '#555';
        ctx.fillText('If nothing appears, check browser permissions (icon in address bar)', 20, 65);
      }

      const curPhase = phaseRef.current;

      /* ── SETUP overlays ── */
      if (curPhase === 'setup') {
        const boxes = setupBoxesRef.current;
        const polys = setupPolygonsRef.current;
        const ds = drawStartRef.current;
        const dc = drawCurrentRef.current;
        const pp = polyPointsRef.current;
        const ns = numSeatsRef.current;
        const curMode = modeRef.current;

        // Completed rect zones
        boxes.forEach((box, i) => {
          ctx.strokeStyle = '#00d4ff'; ctx.lineWidth = 3;
          ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
          ctx.fillStyle = 'rgba(0,212,255,0.1)';
          ctx.fillRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
          ctx.fillStyle = '#00d4ff'; ctx.font = 'bold 20px Inter, sans-serif';
          ctx.fillText(`S${i + 1}`, box[0] + 8, box[1] + 24);
        });

        // Completed polygon zones
        const pColors = ['#00d4ff', '#a78bfa', '#00e676', '#ffaa00', '#ff3d71'];
        polys.forEach((poly, i) => {
          const c = pColors[i % pColors.length];
          ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]);
          poly.slice(1).forEach((p) => ctx.lineTo(p[0], p[1]));
          ctx.closePath(); ctx.strokeStyle = c; ctx.lineWidth = 3; ctx.stroke();
          ctx.fillStyle = c + '1a'; ctx.fill();
          const cx = poly.reduce((s, p) => s + p[0], 0) / 4;
          const cy = poly.reduce((s, p) => s + p[1], 0) / 4;
          ctx.fillStyle = c; ctx.font = 'bold 20px Inter, sans-serif';
          ctx.fillText(`S${i + 1}`, cx - 10, cy + 7);
        });

        // In-progress rect
        if (ds && dc) {
          ctx.strokeStyle = '#ffaa00'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
          ctx.strokeRect(ds.x, ds.y, dc.x - ds.x, dc.y - ds.y);
          ctx.setLineDash([]);
        }

        // In-progress polygon points
        if (pp.length > 0) {
          ctx.fillStyle = '#ffaa00';
          pp.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, Math.PI * 2); ctx.fill(); });
          if (pp.length >= 2) {
            ctx.strokeStyle = '#ffaa00'; ctx.lineWidth = 2; ctx.beginPath();
            ctx.moveTo(pp[0][0], pp[0][1]);
            pp.slice(1).forEach((p) => ctx.lineTo(p[0], p[1]));
            ctx.stroke();
          }
        }

        // Instructions overlay
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.fillRect(0, 0, canvas.width, 40);
        ctx.fillStyle = '#fff'; ctx.font = '14px Inter, sans-serif';
        const done = curMode === 'polygon' ? polys.length : boxes.length;
        const instr = curMode === 'polygon'
          ? `Click 4 corners per seat — ${done}/${ns} done | Points: ${pp.length}/4`
          : `Drag to draw seat zones — ${done}/${ns} done`;
        ctx.fillText(instr, 12, 26);
      }

      /* ── DETECT phase ── */
      if (curPhase === 'detect') {
        const curSeats = seatsRef.current;
        const curOcc = occRef.current;
        const curDets = detsRef.current;
        const curMode = modeRef.current;

        // FPS counter
        fpsCountRef.current++;
        const now = performance.now();
        if (now - fpsTimeRef.current >= 1000) {
          setFps(Math.round(fpsCountRef.current * 1000 / (now - fpsTimeRef.current)));
          fpsCountRef.current = 0;
          fpsTimeRef.current = now;
        }

        // Run inference based on user-selected target FPS.
        // We compute a minimum gap between YOLO runs; busy-lock still prevents overlap.
        const tfps = Math.max(5, Math.min(30, targetFpsRef.current || 15));
        const MIN_INFERENCE_GAP = 1000 / tfps;
        if (
          !detectBusy.current &&
          detectorRef.current?.isReady &&
          videoOk &&
          (now - lastInferenceRef.current) >= MIN_INFERENCE_GAP
        ) {
          detectBusy.current = true;
          lastInferenceRef.current = now;
          detectorRef.current.detect(video, confRef.current)
            .then((newDets) => {
              detsRef.current = newDets; setDetections(newDets);
              const raw = checkOccupancy(newDets, seatsRef.current, modeRef.current);
              const smoothed = smootherRef.current?.update(raw) ?? raw;
              occRef.current = smoothed; setOccupied(smoothed);
            })
            .catch((e) => console.error('Detection error:', e))
            .finally(() => { detectBusy.current = false; });
        }

        // Draw overlays on display canvas

        // Seat zones — thin border + transparent fill
        curSeats.forEach((s, i) => {
          const occ = curOcc[i];
          const color = occ ? '#ff3d71' : '#00e676';
          if (curMode === 'polygon' && s.polygon) {
            ctx.beginPath(); ctx.moveTo(s.polygon[0][0], s.polygon[0][1]);
            s.polygon.slice(1).forEach((p) => ctx.lineTo(p[0], p[1]));
            ctx.closePath(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
            ctx.fillStyle = occ ? 'rgba(255,61,113,0.08)' : 'rgba(0,230,118,0.08)'; ctx.fill();
            const cx = s.polygon.reduce((sum, p) => sum + p[0], 0) / s.polygon.length;
            const cy = s.polygon.reduce((sum, p) => sum + p[1], 0) / s.polygon.length;
            ctx.fillStyle = color; ctx.font = 'bold 14px Inter, sans-serif';
            ctx.fillText(s.id, cx - 10, cy + 5);
          } else if (s.box) {
            const [x1, y1, x2, y2] = s.box;
            ctx.strokeStyle = color; ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = occ ? 'rgba(255,61,113,0.08)' : 'rgba(0,230,118,0.08)';
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = color;
            ctx.font = 'bold 13px Inter, sans-serif';
            ctx.fillText(s.id, x1 + 4, y1 + 16);
          }
        });

        // Person detections — shrunk 40% + dashed + subtle
        const SHRINK = 0.40;
        curDets.forEach((d) => {
          const w = d.x2 - d.x1, h = d.y2 - d.y1;
          const sx1 = d.x1 + w * (SHRINK / 2);
          const sy1 = d.y1 + h * (SHRINK / 2);
          const sw = w * (1 - SHRINK);
          const sh = h * (1 - SHRINK);
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = 'rgba(255,204,0,0.5)'; ctx.lineWidth = 1;
          ctx.strokeRect(sx1, sy1, sw, sh);
          ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(255,204,0,0.6)'; ctx.font = '10px Inter, sans-serif';
          ctx.fillText(`${(d.confidence * 100).toFixed(0)}%`, sx1, Math.max(10, sy1 - 3));
        });
      }

      rafRef.current = requestAnimationFrame(render);
    };

    rafRef.current = requestAnimationFrame(render);
    return () => { active = false; cancelAnimationFrame(rafRef.current); };
  }, [phase]);

  /* ─── Save setup & transition to detect ───── */
  const saveSetup = () => {
    let seatDefs: SeatConfig[];
    if (mode === 'polygon') {
      seatDefs = setupPolygons.map((poly, i) => ({ id: `S${i + 1}`, polygon: poly }));
    } else {
      seatDefs = setupBoxes.map((box, i) => ({ id: `S${i + 1}`, box: box }));
    }
    setSeats(seatDefs);
    // Slightly shorter smoothing window for faster seat-state updates.
    smootherRef.current = new TemporalSmoother(seatDefs.length, 5, 0.4);
    setOccupied(new Array(seatDefs.length).fill(false));
    setPhase('detect');
    loadModel();
  };

  /* ─── Phase transitions ──────────────────── */
  const goToSetup = async () => {
    setSetupBoxes([]); setSetupPolygons([]); setPolyPoints([]);
    setCameraError('');
    try {
      await startCamera();
      setPhase('setup');
    } catch {
      // Error already set in startCamera
    }
  };

  const goToConfigure = () => {
    stopCamera();
    setPhase('configure');
    setDetections([]); setOccupied([]);
  };

  const seatsDrawn = mode === 'polygon' ? setupPolygons.length : setupBoxes.length;
  const canSave = seatsDrawn === numSeats && numSeats > 0;

  /* ──────────────────────────────────────────
     RENDER — single persistent canvas
     ────────────────────────────────────────── */
  return (
    <div className="app-shell">
      {/* Persistent hidden video — never unmounted */}
      <video
        ref={videoRef}
        autoPlay playsInline muted
        style={{ position: 'fixed', top: -9999, left: -9999, width: 1, height: 1, opacity: 0, pointerEvents: 'none' }}
      />

      {/* Header */}
      <header className="app-header glass">
        <h1>Seat Occupancy Monitor</h1>
        <span className="badge">
          {phase === 'configure' ? 'Configure' : phase === 'setup' ? 'Setup' : 'Live Detection'}
        </span>
      </header>

      {/* ─── CONFIGURE ─────────────────────── */}
      {phase === 'configure' && (
        <div className="config-panel glass fade-in" id="config-panel">
          <h2>Configure Detection</h2>
          <p className="subtitle">
            Set up your camera, choose a detection mode, and define how many
            seats to monitor. All AI processing runs locally in your browser.
          </p>

          <div className="form-group">
            <label htmlFor="camera-select">Camera</label>
            <select id="camera-select" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
              {cameras.length === 0 && <option value="">Default camera (grant access to see devices)</option>}
              {cameras.map((c, i) => (
                <option key={c.deviceId} value={c.deviceId}>{c.label || `Camera ${i + 1}`}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Detection Mode</label>
            <div className="mode-grid">
              {(Object.keys(MODE_INFO) as DetectionMode[]).map((m) => (
                <button key={m} className={`mode-card ${mode === m ? 'active' : ''}`} onClick={() => setMode(m)} id={`mode-${m}`}>
                  <span className="icon">{MODE_INFO[m].icon}</span>
                  <span className="name">{MODE_INFO[m].label}</span>
                  <span className="desc">{MODE_INFO[m].desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* ── Number of seats ── */}
          <div className="form-group">
            <label htmlFor="num-seats">Number of seats to monitor</label>
            <div className="seats-input-wrapper">
              <button className="seats-stepper" onClick={() => setNumSeats((n) => Math.max(1, n - 1))} disabled={numSeats <= 1} id="seats-minus" type="button">−</button>
              <input id="num-seats" type="number" min={1} max={20} value={numSeats}
                onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v) && v >= 1 && v <= 20) setNumSeats(v); }}
                className="seats-number-input" />
              <button className="seats-stepper" onClick={() => setNumSeats((n) => Math.min(20, n + 1))} disabled={numSeats >= 20} id="seats-plus" type="button">+</button>
            </div>
            <span className="seats-hint">You&apos;ll draw {numSeats} seat zone{numSeats !== 1 ? 's' : ''} on the camera feed</span>
          </div>

          <div className="form-group">
            <label htmlFor="confidence-slider">Confidence Threshold: {(confidence * 100).toFixed(0)}%</label>
            <input id="confidence-slider" type="range" min={0.1} max={0.9} step={0.05} value={confidence} onChange={(e) => setConfidence(parseFloat(e.target.value))} />
          </div>

          {cameraError && (
            <div className="camera-error" id="camera-error">
              <span className="error-icon">⚠️</span>
              <div>
                <p className="error-msg">{cameraError}</p>
                <button className="btn btn-sm btn-secondary" onClick={() => { setCameraError(''); enumerateCameras(); }} style={{ marginTop: 8 }}>🔄 Re-check Cameras</button>
              </div>
            </div>
          )}

          <button className="btn btn-primary" onClick={goToSetup} id="start-setup-btn">
            📷 Start Camera &amp; Draw Seat Zones
          </button>
        </div>
      )}

      {/* ─── SETUP / DETECT — canvas always present ─── */}
      {(phase === 'setup' || phase === 'detect') && (
        <div className="fade-in">
          {/* Model loader overlay (detect only) */}
          {phase === 'detect' && modelStatus !== 'ready' && (
            <div className="model-loader glass">
              {(modelStatus === 'loading' || modelStatus === 'idle') && (
                <>
                  <div className="spinner" />
                  <div className="progress-bar-outer">
                    <div className="progress-bar-inner" style={{ width: `${modelProgress}%` }} />
                  </div>
                  <p>Loading YOLOv8 model… {modelProgress.toFixed(0)}%</p>
                </>
              )}
              {modelStatus === 'error' && (
                <>
                  <p style={{ color: 'var(--danger)' }}>Failed to load model. Ensure yolov8n.onnx is in public/.</p>
                  <button className="btn btn-primary" onClick={loadModel}>Retry</button>
                </>
              )}
            </div>
          )}

          {/* Camera + single persistent canvas */}
          <div>
            <div className="camera-wrapper">
              <canvas
                ref={canvasRef}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                style={{
                  cursor: phase === 'setup' ? 'crosshair' : 'default',
                  width: '100%', height: '100%',
                }}
              />
              {phase === 'detect' && (
                <div className="camera-overlay-bar">
                  {showFps && <span className="fps-badge">FPS: {fps}</span>}
                  <span className="mode-badge">{mode.toUpperCase()}</span>
                </div>
              )}
            </div>

            {/* Setup toolbar */}
            {phase === 'setup' && (
              <div className="setup-toolbar glass" style={{ marginTop: 12 }}>
                <span className="counter">Seats defined: <strong>{seatsDrawn}</strong> / {numSeats}</span>
                <button className="btn btn-sm btn-secondary" id="undo-btn"
                  onClick={() => {
                    if (mode === 'polygon') {
                      if (polyPoints.length > 0) setPolyPoints([]);
                      else setSetupPolygons((p) => p.slice(0, -1));
                    } else { setSetupBoxes((b) => b.slice(0, -1)); }
                  }}>↩ Undo</button>
                <button className="btn btn-sm btn-danger" id="clear-btn"
                  onClick={() => { setSetupBoxes([]); setSetupPolygons([]); setPolyPoints([]); }}>✕ Clear</button>
                <button className="btn btn-sm btn-primary" disabled={!canSave} onClick={saveSetup} id="save-setup-btn">✓ Save &amp; Start Detection</button>
                <button className="btn btn-sm btn-secondary" onClick={goToConfigure} id="back-btn">← Back</button>
              </div>
            )}

            {/* Detect controls */}
            {phase === 'detect' && (
              <div className="detect-controls glass" style={{ marginTop: 12 }}>
                <div className="slider-group">
                  <span>Confidence:</span>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={confidence}
                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                    id="detect-confidence"
                  />
                  <span className="val">{(confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="slider-group">
                  <span>Target FPS:</span>
                  <input
                    type="range"
                    min={5}
                    max={30}
                    step={1}
                    value={targetFps}
                    onChange={(e) => setTargetFps(parseInt(e.target.value, 10) || 15)}
                    id="detect-fps"
                  />
                  <span className="val">{targetFps} fps</span>
                </div>
                <div className="toggle-row" style={{ gap: 8 }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Show FPS</span>
                  <button
                    className={`toggle ${showFps ? 'on' : ''}`}
                    onClick={() => setShowFps(!showFps)}
                    id="fps-toggle"
                  />
                </div>
                <button className="btn btn-sm btn-secondary" onClick={goToConfigure} style={{ marginLeft: 'auto' }} id="reconfigure-btn">⚙ Reconfigure</button>
              </div>
            )}
          </div>

          {/* ── Bottom seat status strip (detect only) ── */}
          {phase === 'detect' && (
            <div className="bottom-status-strip">
              <div className="status-seats-row">
                {seats.map((s, i) => (
                  <div key={s.id} className={`status-box ${occupied[i] ? 'occupied' : 'free'}`} id={`seat-${s.id}`}>
                    <div className="status-box-dot" />
                    <span className="status-box-id">{s.id}</span>
                    <span className="status-box-label">{occupied[i] ? 'Occupied' : 'Available'}</span>
                  </div>
                ))}
              </div>
              <div className="status-summary">
                <span className="summary-chip free-chip">{occupied.filter((o) => !o).length} Free</span>
                <span className="summary-chip occ-chip">{occupied.filter(Boolean).length} Occupied</span>
                <span className="summary-chip total-chip">{seats.length} Total</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
