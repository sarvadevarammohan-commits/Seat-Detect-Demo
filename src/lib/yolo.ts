import type { Detection } from './types';

const MODEL_INPUT_SIZE = 640;
const PERSON_CLASS_ID = 0;

let ortModule: typeof import('onnxruntime-web') | null = null;

async function getOrt() {
  if (!ortModule) {
    ortModule = await import('onnxruntime-web');
    ortModule.env.wasm.wasmPaths =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';
    ortModule.env.wasm.numThreads = 1;
  }
  return ortModule;
}

interface LetterboxInfo {
  scale: number;
  padX: number;
  padY: number;
}

export class YOLODetector {
  private session: InstanceType<
    typeof import('onnxruntime-web').InferenceSession
  > | null = null;
  private inputName = '';
  private outputName = '';

  async load(
    modelUrl: string,
    onProgress?: (pct: number) => void
  ): Promise<void> {
    const ort = await getOrt();
    const buffer = await this.fetchWithProgress(modelUrl, onProgress);
    this.session = await ort.InferenceSession.create(buffer, {
      executionProviders: ['wasm'],
    });
    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
  }

  get isReady(): boolean {
    return this.session !== null;
  }

  private async fetchWithProgress(
    url: string,
    onProgress?: (pct: number) => void
  ): Promise<ArrayBuffer> {
    const resp = await fetch(url);
    const total = parseInt(resp.headers.get('content-length') || '0', 10);
    const reader = resp.body!.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress?.(total ? (received / total) * 100 : 50);
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) {
      buf.set(c, off);
      off += c.length;
    }
    return buf.buffer;
  }

  private preprocess(
    imageData: ImageData,
    targetSize = MODEL_INPUT_SIZE
  ): { tensor: Float32Array; letterbox: LetterboxInfo } {
    const { width, height } = imageData;
    const scale = Math.min(targetSize / width, targetSize / height);
    const newW = Math.round(width * scale);
    const newH = Math.round(height * scale);
    const padX = Math.round((targetSize - newW) / 2);
    const padY = Math.round((targetSize - newH) / 2);

    const srcCanvas = new OffscreenCanvas(width, height);
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCtx.putImageData(imageData, 0, 0);

    const canvas = new OffscreenCanvas(targetSize, targetSize);
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, targetSize, targetSize);
    ctx.drawImage(srcCanvas, padX, padY, newW, newH);

    const resized = ctx.getImageData(0, 0, targetSize, targetSize);
    const pixels = resized.data;
    const cSize = targetSize * targetSize;
    const tensor = new Float32Array(3 * cSize);
    for (let i = 0; i < cSize; i++) {
      const pi = i * 4;
      tensor[i] = pixels[pi] / 255;
      tensor[cSize + i] = pixels[pi + 1] / 255;
      tensor[2 * cSize + i] = pixels[pi + 2] / 255;
    }
    return { tensor, letterbox: { scale, padX, padY } };
  }

  async detect(
    imageData: ImageData,
    confThreshold = 0.4
  ): Promise<Detection[]> {
    if (!this.session) throw new Error('Model not loaded');
    const ort = await getOrt();
    const { tensor, letterbox } = this.preprocess(imageData);
    const inputTensor = new ort.Tensor('float32', tensor, [
      1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
    ]);
    const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
    feeds[this.inputName] = inputTensor;
    const results = await this.session.run(feeds);
    const output = results[this.outputName];
    const data = output.data as Float32Array;

    // Output: [1, 84, 8400]  —  84 = 4 bbox + 80 classes
    const numDets = 8400;
    const candidates: Detection[] = [];
    for (let i = 0; i < numDets; i++) {
      const score = data[(4 + PERSON_CLASS_ID) * numDets + i];
      if (score < confThreshold) continue;
      const cx = data[0 * numDets + i];
      const cy = data[1 * numDets + i];
      const w = data[2 * numDets + i];
      const h = data[3 * numDets + i];
      let x1 = (cx - w / 2 - letterbox.padX) / letterbox.scale;
      let y1 = (cy - h / 2 - letterbox.padY) / letterbox.scale;
      let x2 = (cx + w / 2 - letterbox.padX) / letterbox.scale;
      let y2 = (cy + h / 2 - letterbox.padY) / letterbox.scale;
      x1 = Math.max(0, x1);
      y1 = Math.max(0, y1);
      x2 = Math.min(imageData.width, x2);
      y2 = Math.min(imageData.height, y2);
      candidates.push({ x1, y1, x2, y2, confidence: score });
    }
    return this.nms(candidates, 0.5);
  }

  private nms(dets: Detection[], iouThr: number): Detection[] {
    dets.sort((a, b) => b.confidence - a.confidence);
    const kept: Detection[] = [];
    const skip = new Set<number>();
    for (let i = 0; i < dets.length; i++) {
      if (skip.has(i)) continue;
      kept.push(dets[i]);
      for (let j = i + 1; j < dets.length; j++) {
        if (skip.has(j)) continue;
        if (this.iou(dets[i], dets[j]) >= iouThr) skip.add(j);
      }
    }
    return kept;
  }

  private iou(a: Detection, b: Detection): number {
    const x1 = Math.max(a.x1, b.x1),
      y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2),
      y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (inter === 0) return 0;
    const aA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const aB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (aA + aB - inter);
  }
}
