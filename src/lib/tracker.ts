import { Detection } from './types';

export interface TrackedObject extends Detection {
  id: number;
  lastSeen: number;
  framesSeen: number;
}

export class ObjectTracker {
  private objects: TrackedObject[] = [];
  private nextId = 1;
  private maxAge = 15; // Max frames to keep an object alive without detection

  update(newDets: Detection[]): TrackedObject[] {
    const now = Date.now();
    const currentFrameObjects: TrackedObject[] = [];

    // Assign new detections to existing objects (Greedy IoU match)
    const matches = new Map<number, number>(); // detIdx -> objectIdx
    
    newDets.forEach((det, detIdx) => {
      let bestIou = 0.3; // Threshold
      let bestIdx = -1;

      this.objects.forEach((obj, objIdx) => {
        const iou = this.computeIoU(det, obj);
        if (iou > bestIou) {
          bestIou = iou;
          bestIdx = objIdx;
        }
      });

      if (bestIdx !== -1) {
        matches.set(detIdx, bestIdx);
      }
    });

    // Update matched objects
    newDets.forEach((det, detIdx) => {
      if (matches.has(detIdx)) {
        const objIdx = matches.get(detIdx)!;
        const obj = this.objects[objIdx];
        obj.x1 = det.x1;
        obj.y1 = det.y1;
        obj.x2 = det.x2;
        obj.y2 = det.y2;
        obj.confidence = det.confidence;
        obj.lastSeen = now;
        obj.framesSeen++;
        currentFrameObjects.push(obj);
      } else {
        // New object
        const newObj: TrackedObject = {
          ...det,
          id: this.nextId++,
          lastSeen: now,
          framesSeen: 1
        };
        this.objects.push(newObj);
        currentFrameObjects.push(newObj);
      }
    });

    // Cleanup old objects
    this.objects = this.objects.filter(obj => {
      const isPresent = currentFrameObjects.some(o => o.id === obj.id);
      if (isPresent) return true;
      // Keep alive for a bit if not seen
      const age = now - obj.lastSeen;
      return age < 1000; // 1 second grace
    });

    return currentFrameObjects;
  }

  private computeIoU(a: Detection, b: Detection): number {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (inter === 0) return 0;
    const combinedArea = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (combinedArea - inter);
  }
}
