import type { Detection, DetectionMode, SeatConfig } from './types';

const IOU_THRESHOLD = 0.08;

function computeIoU(a: number[], b: number[]): number {
  const xa = Math.max(a[0], b[0]), ya = Math.max(a[1], b[1]);
  const xb = Math.min(a[2], b[2]), yb = Math.min(a[3], b[3]);
  const inter = Math.max(0, xb - xa) * Math.max(0, yb - ya);
  if (inter === 0) return 0;
  const aA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
  const aB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
  return inter / Math.min(aA, aB);
}

function overlapRatio(p: number[], s: number[]): number {
  const xa = Math.max(p[0], s[0]), ya = Math.max(p[1], s[1]);
  const xb = Math.min(p[2], s[2]), yb = Math.min(p[3], s[3]);
  const inter = Math.max(0, xb - xa) * Math.max(0, yb - ya);
  return inter / Math.max(1, (s[2] - s[0]) * (s[3] - s[1]));
}

function centerIn(p: number[], s: number[]): boolean {
  const cx = (p[0] + p[2]) / 2, cy = (p[1] + p[3]) / 2;
  return s[0] <= cx && cx <= s[2] && s[1] <= cy && cy <= s[3];
}

function personInSeat(p: number[], s: number[]): boolean {
  return computeIoU(p, s) >= IOU_THRESHOLD || overlapRatio(p, s) >= 0.2 || centerIn(p, s);
}

function bottomCenter(p: number[]): [number, number] {
  return [(p[0] + p[2]) / 2, p[3]];
}

function anchorInRect(p: number[], s: number[]): boolean {
  const [cx, by] = bottomCenter(p);
  return s[0] <= cx && cx <= s[2] && s[1] <= by && by <= s[3];
}

function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function anchorInPolygon(p: number[], poly: [number, number][]): boolean {
  const [cx, by] = bottomCenter(p);
  return pointInPolygon(cx, by, poly);
}

function centerInPolygon(p: number[], poly: [number, number][]): boolean {
  const cx = (p[0] + p[2]) / 2;
  const cy = (p[1] + p[3]) / 2;
  return pointInPolygon(cx, cy, poly);
}

export function checkOccupancy(
  dets: Detection[], seats: SeatConfig[], mode: DetectionMode
): boolean[] {
  if (mode === 'exclusive') return exclusive(dets, seats);
  const occ = new Array(seats.length).fill(false);
  for (let i = 0; i < seats.length; i++) {
    const s = seats[i];
    for (const d of dets) {
      const pb = [d.x1, d.y1, d.x2, d.y2];
      let match = false;

      if (mode === 'polygon') {
        if (s.polygon) {
          // Check anchor point in polygon OR center in polygon
          match = anchorInPolygon(pb, s.polygon) || centerInPolygon(pb, s.polygon);
        } else if (s.box) {
          // Fallback: treat box as polygon
          match = personInSeat(pb, s.box);
        }
      } else {
        // For 'anchor' and 'rectangle' modes: use HYBRID check
        // This works for both eye-level cameras AND elevated CCTV
        if (s.box) {
          match = personInSeat(pb, s.box) || anchorInRect(pb, s.box);
        }
      }

      if (match) { occ[i] = true; break; }
    }
  }
  return occ;
}

function exclusive(dets: Detection[], seats: SeatConfig[]): boolean[] {
  const occ = new Array(seats.length).fill(false);
  for (const d of dets) {
    const pb = [d.x1, d.y1, d.x2, d.y2];
    let bestIdx = -1, bestScore = 0;
    for (let i = 0; i < seats.length; i++) {
      if (!seats[i].box) continue;
      const sc = Math.max(overlapRatio(pb, seats[i].box!), computeIoU(pb, seats[i].box!));
      if (sc > bestScore && sc >= IOU_THRESHOLD) { bestScore = sc; bestIdx = i; }
    }
    if (bestIdx >= 0) occ[bestIdx] = true;
  }
  return occ;
}

export class TemporalSmoother {
  private hist: boolean[][];
  private win: number;
  private ratio: number;

  constructor(n: number, win = 5, ratio = 0.5) {
    this.win = win;
    this.ratio = ratio;
    this.hist = Array.from({ length: n }, () => []);
  }

  update(raw: boolean[]): boolean[] {
    return raw.map((v, i) => {
      if (i >= this.hist.length) this.hist.push([]);
      this.hist[i].push(v);
      if (this.hist[i].length > this.win) this.hist[i].shift();
      const votes = this.hist[i].filter(Boolean).length;
      return votes >= this.hist[i].length * this.ratio;
    });
  }

  reset(n: number) {
    this.hist = Array.from({ length: n }, () => []);
  }
}
