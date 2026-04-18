import type { Detection, DetectionMode, SeatConfig } from './types';

const IOU_THRESHOLD = 0.05;    // Lower threshold for partial entry
const OVERLAP_THRESHOLD = 0.15; // Trigger with just 15% overlap

function computeIoUUnder(a: number[], b: number[]): number {
  const xa = Math.max(a[0], b[0]), ya = Math.max(a[1], b[1]);
  const xb = Math.min(a[2], b[2]), yb = Math.min(a[3], b[3]);
  const inter = Math.max(0, xb - xa) * Math.max(0, yb - ya);
  if (inter === 0) return 0;
  const aArea = (a[2] - a[0]) * (a[3] - a[1]);
  const bArea = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / Math.min(aArea, bArea);
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

/**
 * NEW: Greedy assignment check.
 * Each detection can only occupy ONE seat (the one with the highest "score").
 * This prevents a single person from triggering multiple adjacent seats.
 */
export function checkOccupancy(
  dets: Detection[], seats: SeatConfig[], mode: DetectionMode
): boolean[] {
  const occ = new Array(seats.length).fill(false);
  if (dets.length === 0 || seats.length === 0) return occ;

  // 1. Calculate all possible scores
  const assignments: { detIdx: number; seatIdx: number; score: number }[] = [];

  dets.forEach((d, dIdx) => {
    const pb = [d.x1, d.y1, d.x2, d.y2];
    seats.forEach((s, sIdx) => {
      let score = 0;
      if (mode === 'polygon' && s.polygon) {
        const anchorInside = anchorInPolygon(pb, s.polygon);
        const centerInside = centerInPolygon(pb, s.polygon);
        // Polygon mode should be strict to avoid neighbor seat bleed.
        score = anchorInside ? 2 + (centerInside ? 0.5 : 0) : 0;
      } else if (s.box) {
        const iou = computeIoUUnder(pb, s.box);
        const ratio = overlapRatio(pb, s.box);
        const centerInside = centerIn(pb, s.box);
        const anchorInside = anchorInRect(pb, s.box);

        if (mode === 'anchor') {
          // Anchor mode: High-angle optimization
          // Sensitive but precise: anchor must be inside OR high overlap ratio
          if (anchorInside) {
            score = 5.0 + ratio * 2.0; // Strong anchor bias
          } else if (centerInside && ratio > 0.3) {
            score = 2.0 + ratio;
          } else if (ratio > 0.45) {
            score = 1.0 + ratio;
          }
        } else if (mode === 'exclusive') {
          const isMatch = iou >= IOU_THRESHOLD || ratio >= 0.5 || centerInside;
          if (isMatch) {
            // Favor 50% overlap and center-of-box heavily
            score = (ratio >= 0.5 ? 10.0 : ratio * 2.0) + (centerInside ? 5.0 : 0) + iou;
          }
        } else {
          // Rectangle mode: Maximum sensitivity
          const isMatch = (iou >= IOU_THRESHOLD || ratio >= OVERLAP_THRESHOLD) && (centerInside || anchorInside);
          if (isMatch) {
            score = Math.max(iou, ratio) * 2.0 + (centerInside ? 1.5 : 0) + (anchorInside ? 1.5 : 0);
          }
        }
      }

      if (score > 0) {
        assignments.push({ detIdx: dIdx, seatIdx: sIdx, score });
      }
    });
  });

  // 2. Sort by score descending and assign greedily
  assignments.sort((a, b) => b.score - a.score);

  const usedDets = new Set<number>();
  const usedSeats = new Set<number>();

  for (const entry of assignments) {
    if (!usedDets.has(entry.detIdx) && !usedSeats.has(entry.seatIdx)) {
      occ[entry.seatIdx] = true;
      usedDets.add(entry.detIdx);
      usedSeats.add(entry.seatIdx);
    }
  }

  return occ;
}

export class TemporalSmoother {
  private hist: boolean[][];
  private win: number;
  private ratio: number;

  constructor(n: number, win = 10, ratio = 0.5) {
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
