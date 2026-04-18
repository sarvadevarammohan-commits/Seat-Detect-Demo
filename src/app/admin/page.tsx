'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import type { AppPhase, DetectionMode, SeatConfig, Detection } from '@/lib/types';
import { MODE_INFO } from '@/lib/types';
import { YOLODetector } from '@/lib/yolo';
import { checkOccupancy, TemporalSmoother } from '@/lib/occupancy';
import { useAuth } from '@/lib/auth-context';
import { db } from '@/lib/firebase';
import { ref, set, update } from 'firebase/database';
import { CameraStabilizer, Matrix } from '@/lib/stabilizer';
import { ObjectTracker, TrackedObject } from '@/lib/tracker';

/* ================================================================
   MAIN PAGE — state machine: configure → setup → detect
   Single persistent canvas & video to avoid ref lifecycle issues.
   ================================================================ */
export default function AdminPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  const [phase, setPhase] = useState<AppPhase>('configure');
  const [mode, setMode] = useState<DetectionMode>('anchor');
  const [numSeats, setNumSeats] = useState(9);
  const [seats, setSeats] = useState<SeatConfig[]>([]);
  const [confidence, setConfidence] = useState(0.40); // Slightly more sensitive
  const [targetFps, setTargetFps] = useState(15);
  const [showFps, setShowFps] = useState(true);
  const [cameraId, setCameraId] = useState('');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState('');

  // Frozen frame state for setup
  const [isFrozen, setIsFrozen] = useState(false);
  const [frozenFrame, setFrozenFrame] = useState<ImageBitmap | null>(null);
  const [freezeTimer, setFreezeTimer] = useState<number | null>(null);

  // Advanced Tracking State
  const [dynamicMode, setDynamicMode] = useState(false);
  const [selectedSeatIdx, setSelectedSeatIdx] = useState<number | null>(null);
  const stabilizerRef = useRef<CameraStabilizer>(new CameraStabilizer());
  const trackerRef = useRef<ObjectTracker>(new ObjectTracker());
  const lastSyncRef = useRef(0);
  const [personTracks, setPersonTracks] = useState<TrackedObject[]>([]);

  // Input source state
  const [inputSource, setInputSource] = useState<'webcam' | 'file'>('webcam');
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Model state
  const [modelStatus, setModelStatus] = useState<
    'idle' | 'loading' | 'ready' | 'error'
  >('idle');
  const [modelProgress, setModelProgress] = useState(0);
  const detectorRef = useRef<YOLODetector | null>(null);

  // Manual Adjustment State
  const [adjustingSeatIdx, setAdjustingSeatIdx] = useState<number | null>(null);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number } | null>(null);

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

  const isFrozenRef = useRef(isFrozen);
  const frozenFrameRef = useRef(frozenFrame);
  useEffect(() => { isFrozenRef.current = isFrozen; }, [isFrozen]);
  useEffect(() => { frozenFrameRef.current = frozenFrame; }, [frozenFrame]);

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

  /* ─── setup input source ─────────────────── */
  const initSource = useCallback(async () => {
    setCameraError('');
    const video = videoRef.current;
    if (!video) return;

    // Cleanup previous
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    video.srcObject = null;
    if (video.src) {
      URL.revokeObjectURL(video.src);
      video.removeAttribute('src');
    }

    try {
      if (inputSource === 'webcam') {
        const constraints: MediaStreamConstraints = {
          video: cameraId
            ? { deviceId: { exact: cameraId }, width: { ideal: 640 }, height: { ideal: 480 } }
            : { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        video.srcObject = stream;
        video.loop = false;
      } else {
        if (!fileUrl) throw new Error('No file selected');
        video.src = fileUrl;
        video.loop = true; // Loop videos for detection
      }

      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Media source timed out')), 10000);
        const onPlaying = () => {
          clearTimeout(timeout);
          video.removeEventListener('playing', onPlaying);
          resolve();
        };
        video.addEventListener('playing', onPlaying);
        video.play().catch(reject);
      });
      setCameraReady(true);
      
      if (inputSource === 'webcam') {
        // Re-enumerate to get labels after permission
        const devs = await navigator.mediaDevices.enumerateDevices();
        const vids = devs.filter((d) => d.kind === 'videoinput');
        setCameras(vids);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      console.error('Source error:', err);
      if (msg.includes('Permission') || msg.includes('NotAllowed')) {
        setCameraError('Camera permission denied. Please allow camera access in your browser settings and reload.');
      } else if (msg.includes('NotFound') || msg.includes('DevicesNotFound')) {
        setCameraError('No camera found. Please connect a camera and try again.');
      } else if (msg.includes('NotReadable') || msg.includes('TrackStartError')) {
        setCameraError('Camera is in use by another application. Close other apps using the camera and try again.');
      } else {
        setCameraError(`Source error: ${msg}. Try again or reload.`);
      }
      setCameraReady(false);
      throw err;
    }
  }, [cameraId, inputSource, fileUrl]);

  const stopSource = useCallback(() => {
    runningRef.current = false;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
        video.srcObject = null;
        if (video.src) {
            URL.revokeObjectURL(video.src);
            video.removeAttribute('src');
        }
    }
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
    const { x, y } = getCanvasCoords(e);
    
    // 1. SETUP PHASE: Hit detection for individual seat selection
    if (phase === 'setup' && mode === 'rectangle') {
        const hitIdx = setupBoxes.findIndex(b => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]);
        if (hitIdx !== -1) {
            setSelectedSeatIdx(hitIdx);
            return; // Selection mode, don't start drawing
        }
    }
    
    // 2. DETECT PHASE: Double click to adjust manually
    if (phase === 'detect') {
        const hitIdx = seats.findIndex(s => {
            if (s.box) {
                return x >= s.box[0] && x <= s.box[2] && y >= s.box[1] && y <= s.box[3];
            }
            if (s.polygon) {
                // Simplified hit detection for polygons: center-ish
                const cx = s.polygon.reduce((sum, p) => sum + p[0], 0) / s.polygon.length;
                const cy = s.polygon.reduce((sum, p) => sum + p[1], 0) / s.polygon.length;
                return Math.abs(x - cx) < 30 && Math.abs(y - cy) < 30;
            }
            return false;
        });

        if (hitIdx !== -1 && e.detail === 2) {
            setAdjustingSeatIdx(hitIdx);
            const seat = seats[hitIdx];
            if (seat.box) {
                setDragOffset({ x: x - seat.box[0], y: y - seat.box[1] });
            } else if (seat.polygon) {
                setDragOffset({ x: x - seat.polygon[0][0], y: y - seat.polygon[0][1] });
            }
            return;
        }
    }

    // 3. Clear selection if clicking empty space
    setSelectedSeatIdx(null);
    
    // 4. Normal drawing logic (Setup only)
    if (phase === 'setup') {
      if (mode === 'polygon') {
        const newPts = [...polyPoints, [x, y] as [number, number]];
        if (newPts.length === 4) {
          if (setupPolygons.length < numSeats) setSetupPolygons((prev) => [...prev, newPts]);
          setPolyPoints([]);
        } else {
          setPolyPoints(newPts);
        }
      } else {
        if (setupBoxes.length >= numSeats) return;
        setDrawStart({ x, y });
        setDrawCurrent({ x, y });
      }
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getCanvasCoords(e);
    
    if (phaseRef.current === 'setup') {
        if (mode !== 'polygon' && drawStart) setDrawCurrent({ x, y });
    } else if (phaseRef.current === 'detect' && adjustingSeatIdx !== null && dragOffset) {
        setSeats((prev: SeatConfig[]) => {
            const next = [...prev];
            const seat = next[adjustingSeatIdx];
            if (seat.box) {
                const w = seat.box[2] - seat.box[0];
                const h = seat.box[3] - seat.box[1];
                const nx1 = x - dragOffset.x;
                const ny1 = y - dragOffset.y;
                next[adjustingSeatIdx] = { ...seat, box: [nx1, ny1, nx1 + w, ny1 + h] };
            } else if (seat.polygon) {
                const dx = x - dragOffset.x - seat.polygon[0][0];
                const dy = y - dragOffset.y - seat.polygon[0][1];
                next[adjustingSeatIdx] = { 
                    ...seat, 
                    polygon: seat.polygon.map((p: [number, number]) => [p[0] + dx, p[1] + dy] as [number, number]) 
                };
                setDragOffset({ x: x - (seat.polygon[0][0] + dx), y: y - (seat.polygon[0][1] + dy) });
            }
            return next;
        });
    }
  };

  const handleCanvasMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phaseRef.current === 'setup') {
      if (mode !== 'polygon' && drawStart) {
        const end = getCanvasCoords(e);
        const x1 = Math.min(drawStart.x, end.x), y1 = Math.min(drawStart.y, end.y);
        const x2 = Math.max(drawStart.x, end.x), y2 = Math.max(drawStart.y, end.y);
        if (x2 - x1 > 10 && y2 - y1 > 10 && setupBoxes.length < numSeats) {
          setSetupBoxes((prev: [number, number, number, number][]) => [...prev, [x1, y1, x2, y2]]);
        }
        setDrawStart(null);
        setDrawCurrent(null);
      }
    } else if (phaseRef.current === 'detect' && adjustingSeatIdx !== null) {
        // Sync final position to Firebase
        const updatedSeat = seats[adjustingSeatIdx];
        update(ref(db, `seats/${updatedSeat.id}`), updatedSeat)
          .then(() => console.log(`✅ Seat ${updatedSeat.id} manually updated`))
          .catch(err => console.error("❌ Manual sync failed:", err));
        
        setAdjustingSeatIdx(null);
        setDragOffset(null);
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
      const curPhase = phaseRef.current;
      const videoOk = video.readyState >= 2 && video.videoWidth > 0;

      if (videoOk) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        if (curPhase === 'setup' && isFrozenRef.current && frozenFrameRef.current) {
          ctx.drawImage(frozenFrameRef.current, 0, 0, canvas.width, canvas.height);
        } else {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }
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
          const isSelected = selectedSeatIdx === i;
          ctx.strokeStyle = isSelected ? '#ffaa00' : '#00d4ff'; 
          ctx.lineWidth = isSelected ? 5 : 3;
          ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
          ctx.fillStyle = isSelected ? 'rgba(255, 170, 0, 0.2)' : 'rgba(0,212,255,0.1)';
          ctx.fillRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
          ctx.fillStyle = isSelected ? '#ffaa00' : '#00d4ff'; 
          ctx.font = 'bold 20px Inter, sans-serif';
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
            .then((newDets: Detection[]) => {
              // Filter for persons (Class 0)
              const persons = newDets.filter((d: Detection) => d.classId === 0);
              
              // 1. Dynamic Tracking & Stabilization
              if (dynamicMode && video) {
                // Update person IDs
                const tracks = trackerRef.current.update(persons);
                setPersonTracks(tracks);

                // Compute camera stabilization
                const M = stabilizerRef.current.computeTransform(video);
                if (M) {
                  // Adjust seat zones based on camera shift
                  const updatedSeats = seatsRef.current.map(s => {
                    if (s.box) {
                      const [x1, y1] = CameraStabilizer.transformPoint(s.box[0], s.box[1], M);
                      const [x2, y2] = CameraStabilizer.transformPoint(s.box[2], s.box[3], M);
                      return { ...s, box: [x1, y1, x2, y2] as [number, number, number, number] };
                    }
                    if (s.polygon) {
                      return { 
                        ...s, 
                        polygon: s.polygon.map(p => CameraStabilizer.transformPoint(p[0], p[1], M)) as [number, number][] 
                      };
                    }
                    return s;
                  });
                  
                  // Disable auto-tracking for seat being manually adjusted
                  const seatsToUpdate = updatedSeats.map((s, idx) => 
                    idx === adjustingSeatIdx ? seatsRef.current[idx] : s
                  );
                  
                  // Only set state locally to avoid jitter, sync to Firebase periodically
                  setSeats(seatsToUpdate);
                  seatsRef.current = seatsToUpdate;

                  // Sync to Firebase every 2 seconds if movement occurred to keep public dashboard updated
                  const now = Date.now();
                  if (now - lastSyncRef.current > 2000) {
                    lastSyncRef.current = now;
                    const seatsMap: Record<string, SeatConfig> = {};
                    updatedSeats.forEach(s => seatsMap[s.id] = s);
                    update(ref(db), { 'seats': seatsMap });
                  }
                }
              } else {
                setPersonTracks(persons.map((p, i) => ({ ...p, id: i, lastSeen: Date.now(), framesSeen: 1 })));
              }

              setDetections(persons);
              detsRef.current = persons;
              
              const raw = checkOccupancy(persons, seatsRef.current, modeRef.current);
              const smoothed = smootherRef.current?.update(raw) ?? raw;
              
              // Only update if changed to minimize Firebase writes
              const prevOcc = occRef.current;
              const hasChanged = smoothed.some((v, i) => v !== prevOcc[i]);
              
              if (hasChanged) {
                occRef.current = smoothed; 
                setOccupied(smoothed);
                
                // Sync to Firebase Realtime DB
                const statusMap: Record<string, boolean> = {};
                seatsRef.current.forEach((s, i) => {
                  statusMap[s.id] = smoothed[i];
                });
                set(ref(db, 'status'), statusMap)
                  .then(() => console.log('✅ Status synced to Firebase'))
                  .catch((err) => console.error('❌ Status sync failed:', err));
              }
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

          // Manual adjustment highlight
          if (adjustingSeatIdx === i) {
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            if (s.box) {
                const [x1, y1, x2, y2] = s.box;
                ctx.strokeRect(x1 - 2, y1 - 2, (x2 - x1) + 4, (y2 - y1) + 4);
            } else if (s.polygon) {
                ctx.beginPath(); ctx.moveTo(s.polygon[0][0]-2, s.polygon[0][1]-2);
                s.polygon.slice(1).forEach((p) => ctx.lineTo(p[0]-2, p[1]-2));
                ctx.closePath(); ctx.stroke();
            }
            ctx.setLineDash([]);
          }
        });

        // Person detections — premium compact 'bracket' style
        curDets.forEach((d) => {
          const w = d.x2 - d.x1;
          const h = d.y2 - d.y1;
          
          // Cap drawing aspect ratio to prevent 'long' boxes from crossing into empty space
          // Effective person box for drawing is capped at 1:2.5 height:width ratio if it's too tall
          const drawH = Math.min(h, w * 2.5);
          const drawY1 = d.y1 + (h - drawH) / 2;
          
          // Draw a more subtle, high-end bounding box
          const bW = w * 0.7; // compact width
          const bH = drawH * 0.7; // compact height
          const bx1 = d.x1 + (w - bW) / 2;
          const by1 = drawY1 + (drawH - bH) / 2;
          
          // Corner brackets instead of a full rectangle
          const len = Math.min(bW, bH) * 0.25;
          ctx.strokeStyle = 'rgba(255, 204, 0, 0.8)';
          ctx.lineWidth = 2;
          
          // Top-left
          ctx.beginPath(); ctx.moveTo(bx1, by1 + len); ctx.lineTo(bx1, by1); ctx.lineTo(bx1 + len, by1); ctx.stroke();
          // Top-right
          ctx.beginPath(); ctx.moveTo(bx1 + bW - len, by1); ctx.lineTo(bx1 + bW, by1); ctx.lineTo(bx1 + bW, by1 + len); ctx.stroke();
          // Bottom-right
          ctx.beginPath(); ctx.moveTo(bx1 + bW, by1 + bH - len); ctx.lineTo(bx1 + bW, by1 + bH); ctx.lineTo(bx1 + bW - len, by1 + bH); ctx.stroke();
          // Bottom-left
          ctx.beginPath(); ctx.moveTo(bx1 + len, by1 + bH); ctx.lineTo(bx1, by1 + bH); ctx.lineTo(bx1, by1 + bH - len); ctx.stroke();

          // Subtle fill
          ctx.fillStyle = 'rgba(255, 204, 0, 0.05)';
          ctx.fillRect(bx1, by1, bW, bH);

          // Confidence badge
          ctx.fillStyle = 'rgba(255, 204, 0, 0.9)';
          ctx.font = 'bold 10px Inter, sans-serif';
          const track = personTracks.find(t => t.x1 === d.x1 && t.y1 === d.y1);
          const label = track ? `P${track.id} (${(d.confidence * 100).toFixed(0)}%)` : `${(d.confidence * 100).toFixed(0)}%`;
          ctx.fillText(label, bx1, Math.max(10, by1 - 5));
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
    
    // Set reference frame for stabilization
    const video = videoRef.current;
    if (video) stabilizerRef.current.setReferenceFrame(video);
    
    // Sync seat config to Firebase
    const seatsMap: Record<string, SeatConfig> = {};
    seatDefs.forEach(s => seatsMap[s.id] = s);
    set(ref(db, 'seats'), seatsMap)
      .then(() => console.log('✅ Seats synced to Firebase'))
      .catch((err) => console.error('❌ Seats sync failed:', err));
    
    // Sync inputSource to Firebase for dynamic layout
    set(ref(db, 'config/source'), inputSource)
      .then(() => console.log('✅ Input source synced to Firebase'))
      .catch(err => console.error('❌ Source sync failed:', err));
    
    // Initialize status on Firebase
    const initialStatus: Record<string, boolean> = {};
    seatDefs.forEach(s => initialStatus[s.id] = false);
    set(ref(db, 'status'), initialStatus)
      .then(() => console.log('✅ Status initialized on Firebase'))
      .catch((err) => console.error('❌ Status initialization failed:', err));

    // Slightly shorter smoothing window for faster seat-state updates.
    smootherRef.current = new TemporalSmoother(seatDefs.length, 5, 0.4);
    setOccupied(new Array(seatDefs.length).fill(false));
    setPhase('detect');
    loadModel();
  };

  /* ─── Capture & Auto-Detect ──────────────── */
  const freezeCurrentFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    try {
      const bitmap = await createImageBitmap(video);
      setFrozenFrame(bitmap);
      setIsFrozen(true);
      setFreezeTimer(null);
    } catch (e) {
      console.error('Failed to capture frame:', e);
    }
  }, []);

  const unfreeze = () => {
    setIsFrozen(false);
    setFrozenFrame(null);
    setFreezeTimer(null);
  };


  const applyNineSeatTemplate = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = canvas.width;
    const h = canvas.height;

    // Pre-calculated coordinates for image_2.png perspective (Normalized to canvas size)
    const template: [number, number, number, number][] = [
      [0.37 * w, 0.40 * h, 0.47 * w, 0.50 * h], // S1 - Back Left
      [0.48 * w, 0.40 * h, 0.58 * w, 0.50 * h], // S2 - Back Mid
      [0.60 * w, 0.40 * h, 0.70 * w, 0.50 * h], // S3 - Back Right
      [0.05 * w, 0.82 * h, 0.22 * w, 0.98 * h], // S4 - Front Left (Black)
      [0.32 * w, 0.85 * h, 0.48 * w, 1.00 * h], // S5 - Front Mid (Black)
      [0.58 * w, 0.82 * h, 0.75 * w, 0.98 * h], // S6 - Front Right (Black)
      [0.82 * w, 0.32 * h, 0.98 * w, 0.45 * h], // S7 - Side Top
      [0.82 * w, 0.48 * h, 0.98 * w, 0.61 * h], // S8 - Side Mid
      [0.82 * w, 0.64 * h, 0.98 * w, 0.77 * h], // S9 - Side Bottom
    ];

    setSetupBoxes(template.slice(0, numSeats));
    setIsFrozen(true); // Auto-freeze when applying template for clarity
  };

  const autoDetectChairs = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Initialize detector if it doesn't exist yet (usually created in detect phase)
    if (!detectorRef.current) {
      detectorRef.current = new YOLODetector();
    }
    
    setModelStatus('loading');
    try {
      if (!detectorRef.current.isReady) {
        await detectorRef.current.load('/yolov8n.onnx');
      }
      setModelStatus('ready');

      // Use the CANVAS as the source (important for Frozen Reference frames)
      const dets = await detectorRef.current.detect(canvas, 0.15); // Aggressive threshold
    const chairs = dets.filter(d => d.classId === 56);
    const persons = dets.filter(d => d.classId === 0);
    
    // Combine chairs and persons (infer chairs under persons)
    const candidates: [number, number, number, number][] = chairs.map(c => [c.x1, c.y1, c.x2, c.y2]);
    
    persons.forEach(p => {
        // If no chair is registered near this person's lower half, create one
        const hasOverlay = candidates.some(c => {
            const interX = Math.max(c[0], p.x1) < Math.min(c[2], p.x2);
            const interY = Math.max(c[1], p.y1) < Math.min(c[3], p.y2);
            return interX && interY;
        });
        if (!hasOverlay) {
            // Predict chair zone: same width as person, but centered on their lower half
            const w = p.x2 - p.x1;
            const h = p.y2 - p.y1;
            candidates.push([p.x1, p.y1 + h * 0.4, p.x2, p.y2]);
        }
    });

    if (candidates.length === 0) {
      alert("No chairs or occupants detected. Adjust camera or use manual setup.");
      return;
    }

      setSetupBoxes(candidates);
      setNumSeats(candidates.length);
      setIsFrozen(true);
    } catch (err) {
      console.error("Auto-detect failed:", err);
      setModelStatus('error');
      alert("AI Detection failed. Please ensure yolov8n.onnx is accessible.");
    }
  };

  const removeSelectedSeat = () => {
    if (selectedSeatIdx === null) return;
    setSetupBoxes((prev: [number, number, number, number][]) => {
        const next = prev.filter((_: unknown, i: number) => i !== selectedSeatIdx);
        setNumSeats(next.length);
        return next;
    });
    setSelectedSeatIdx(null);
  };

  /* ─── Phase transitions ──────────────────── */
  const goToSetup = async () => {
    setSetupBoxes([]); setSetupPolygons([]); setPolyPoints([]);
    setCameraError('');
    unfreeze();
    try {
      await initSource();
      setPhase('setup');
      setFreezeTimer(5); // Start 5s countdown
    } catch {
      // Error already set in initSource
    }
  };

  // Handle freeze countdown
  useEffect(() => {
    if (phase === 'setup' && freezeTimer !== null) {
      if (freezeTimer > 0) {
        const t = setTimeout(() => setFreezeTimer(freezeTimer - 1), 1000);
        return () => clearTimeout(t);
      } else {
        freezeCurrentFrame();
      }
    }
  }, [phase, freezeTimer, freezeCurrentFrame]);

  const goToConfigure = () => {
    stopSource();
    setPhase('configure');
    setDetections([]); setOccupied([]);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      setFileUrl(URL.createObjectURL(file));
      setInputSource('file');
    }
  };

  const seatsDrawn = mode === 'polygon' ? setupPolygons.length : setupBoxes.length;
  const canSave = seatsDrawn === numSeats && numSeats > 0;

  /* ──────────────────────────────────────────
     RENDER — single persistent canvas
     ────────────────────────────────────────── */
  if (authLoading || !user) {
    return (
      <div className="app-shell flex items-center justify-center min-h-screen">
        <div className="spinner" />
      </div>
    );
  }

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
            Focus on the <strong>Visible First Block</strong> around the central table. 
            Avoid seats attached to the wall as they are prone to occlusion. 
            Detection runs locally in your browser.
          </p>

          <div className="form-group">
            <label>Input Source</label>
            <div className="source-toggle">
              <button 
                className={`toggle-btn ${inputSource === 'webcam' ? 'active' : ''}`}
                onClick={() => setInputSource('webcam')}
                type="button"
              >
                📷 Webcam
              </button>
              <button 
                className={`toggle-btn ${inputSource === 'file' ? 'active' : ''}`}
                onClick={() => {
                  setInputSource('file');
                  if (!fileUrl) fileInputRef.current?.click();
                }}
                type="button"
              >
                📁 Upload Photo/Video
              </button>
            </div>
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept="video/*,image/*" 
              style={{ display: 'none' }} 
            />
            {inputSource === 'file' && fileUrl && (
              <div className="file-preview">
                <span className="file-name">✅ File ready</span>
                <button className="btn-text" onClick={() => fileInputRef.current?.click()}>Change file</button>
              </div>
            )}
          </div>

          {inputSource === 'webcam' && (
            <div className="form-group">
              <label htmlFor="camera-select">Camera</label>
              <select id="camera-select" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
                {cameras.length === 0 && <option value="">Default camera (grant access to see devices)</option>}
                {cameras.map((c, i) => (
                  <option key={c.deviceId} value={c.deviceId}>{c.label || `Camera ${i + 1}`}</option>
                ))}
              </select>
            </div>
          )}

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
            <span className="seats-hint">You&apos;ll draw {numSeats} seat zone{numSeats !== 1 ? 's' : ''} on the feed</span>
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
                <button className="btn btn-sm btn-secondary" onClick={() => { setCameraError(''); enumerateCameras(); }} style={{ marginTop: 8 }}>🔄 Retry</button>
              </div>
            </div>
          )}

          <button 
            className="btn btn-primary" 
            onClick={goToSetup} 
            id="start-setup-btn"
            disabled={inputSource === 'file' && !fileUrl}
          >
            {inputSource === 'webcam' ? '📷 Start Camera' : '🚀 Process Uploaded File'} &amp; Draw Zones
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
                  cursor: phase === 'setup' ? 'crosshair' : adjustingSeatIdx !== null ? 'grabbing' : 'default',
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
              <div className="setup-toolbar glass flex-wrap" style={{ marginTop: 12, gap: 8 }}>
                <div className="flex items-center gap-3 mr-auto">
                    <span className="counter">Seats defined: <strong>{seatsDrawn}</strong> / {numSeats}</span>
                    {seatsDrawn > 0 && <span style={{ fontSize: '0.75rem', color: '#888' }}>(Click a box to select/remove)</span>}
                    {freezeTimer !== null && <span className="badge badge-warning">📸 Freezing in {freezeTimer}s...</span>}
                    {isFrozen && <span className="badge badge-success">❄️ Frozen Reference</span>}
                </div>
                
                <div className="flex items-center gap-2">
                    {selectedSeatIdx !== null && (
                      <button className="btn btn-sm btn-danger" onClick={removeSelectedSeat}>
                         🗑️ Remove S{selectedSeatIdx + 1}
                      </button>
                    )}
                    <button className="btn btn-sm btn-primary" onClick={autoDetectChairs} title="Aggressively detect chairs even when occupied">
                         🤖 Auto-Detect
                    </button>
                    <button className="btn btn-sm btn-secondary" onClick={applyNineSeatTemplate}>
                         ✨ 9-Seat Template
                    </button>
                    <button className="btn btn-sm btn-secondary" onClick={isFrozen ? unfreeze : freezeCurrentFrame}>
                        {isFrozen ? '▶ Unfreeze/Live' : '📸 Freeze Reference'}
                    </button>
                    <div className="v-divider" />
                    <button className="btn btn-sm btn-secondary" id="undo-btn"
                    onClick={() => {
                        if (mode === 'polygon') {
                        if (polyPoints.length > 0) setPolyPoints([]);
                        else setSetupPolygons((p) => p.slice(0, -1));
                        } else { setSetupBoxes((b) => b.slice(0, -1)); }
                    }}>↩ Undo</button>
                    <button className="btn btn-sm btn-danger" id="clear-btn"
                    onClick={() => { setSetupBoxes([]); setSetupPolygons([]); setPolyPoints([]); }}>✕ Clear</button>
                    <button className="btn btn-sm btn-primary" disabled={!canSave} onClick={saveSetup} id="save-setup-btn">✓ Save &amp; Start</button>
                    <button className="btn btn-sm btn-secondary" onClick={goToConfigure} id="back-btn">← Back</button>
                </div>
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
                <div className="toggle-row flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Show FPS</span>
                    <button
                        className={`toggle ${showFps ? 'on' : ''}`}
                        onClick={() => setShowFps(!showFps)}
                        id="fps-toggle"
                    />
                  </div>
                  <div className="v-divider" />
                  <button className="btn btn-sm btn-secondary" onClick={goToConfigure} id="reconfigure-btn">⚙ Reconfigure</button>
                </div>
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
