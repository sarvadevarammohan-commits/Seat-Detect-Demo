import type { Detection } from './types';
import type { InferenceSession, Tensor } from 'onnxruntime-web';

const MODEL_INPUT_SIZE = 640;
const PERSON_CLASS_ID = 0;

let ortModule: typeof import('onnxruntime-web') | null = null;

async function getOrt() {
  if (!ortModule) {
    ortModule = await import('onnxruntime-web');
    ortModule.env.wasm.wasmPaths =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';
    // Use available CPU threads for higher throughput on WASM backend.
    const hwThreads = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 2 : 2;
    ortModule.env.wasm.numThreads = Math.max(1, Math.min(4, hwThreads));
  }
  return ortModule;
}

interface LetterboxInfo {
  scale: number;
  padX: number;
  padY: number;
}

export class YOLODetector {
  private session: InferenceSession | null = null;
  private inputName = '';
  private outputName = '';

  async load(
    modelUrl: string,
    onProgress?: (pct: number) => void
  ): Promise<void> {
    const ort = await getOrt();
    const buffer = await this.fetchWithProgress(modelUrl, onProgress);
    const providers = typeof navigator !== 'undefined' && 'gpu' in navigator
      ? (['webgpu', 'wasm'] as const)
      : (['wasm'] as const);
    this.session = await ort.InferenceSession.create(buffer, {
      executionProviders: [...providers],
      graphOptimizationLevel: 'all',
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

  private targetCanvas: OffscreenCanvas | null = null;
  private tensorValue: Float32Array | null = null;

  private preprocess(
    source: CanvasImageSource,
    targetSize = MODEL_INPUT_SIZE
  ): { tensor: Float32Array; letterbox: LetterboxInfo } {
    const width = source instanceof HTMLVideoElement ? source.videoWidth : (source as any).width;
    const height = source instanceof HTMLVideoElement ? source.videoHeight : (source as any).height;

    const scale = Math.min(targetSize / width, targetSize / height);
    const newW = Math.round(width * scale);
    const newH = Math.round(height * scale);
    const padX = Math.round((targetSize - newW) / 2);
    const padY = Math.round((targetSize - newH) / 2);

    if (!this.targetCanvas) {
      this.targetCanvas = new OffscreenCanvas(targetSize, targetSize);
    }
    const ctx = this.targetCanvas.getContext('2d', { willReadFrequently: true })!;
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, targetSize, targetSize);
    ctx.drawImage(source, padX, padY, newW, newH);

    const resized = ctx.getImageData(0, 0, targetSize, targetSize);
    const pixels = resized.data;
    const cSize = targetSize * targetSize;
    if (!this.tensorValue) this.tensorValue = new Float32Array(3 * cSize);
    
    for (let i = 0; i < cSize; i++) {
      const pi = i * 4;
      this.tensorValue[i] = pixels[pi] / 255;
      this.tensorValue[cSize + i] = pixels[pi + 1] / 255;
      this.tensorValue[2 * cSize + i] = pixels[pi + 2] / 255;
    }
    return { tensor: this.tensorValue, letterbox: { scale, padX, padY } };
  }

  async detect(
    source: CanvasImageSource,
    confThreshold = 0.4
  ): Promise<Detection[]> {
    if (!this.session) throw new Error('Model not loaded');
    const ort = await getOrt();
    const { tensor, letterbox } = this.preprocess(source);
    const inputTensor = new ort.Tensor('float32', tensor, [
      1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
    ]);
    const feeds: Record<string, Tensor> = {};
    feeds[this.inputName] = inputTensor;
    const results = await this.session.run(feeds);
    const output = results[this.outputName];
    const data = output.data as Float32Array;

    const width = source instanceof HTMLVideoElement ? source.videoWidth : (source as any).width;
    const height = source instanceof HTMLVideoElement ? source.videoHeight : (source as any).height;

    // Output: [1, 84, 8400]
    const numDets = 8400;
    const candidates: Detection[] = [];
    for (let i = 0; i < numDets; i++) {
      const score = data[(4 + PERSON_CLASS_ID) * numDets + i];
      if (score < confThreshold) continue;
      const cxb = data[0 * numDets + i];
      const cyb = data[1 * numDets + i];
      const w = data[2 * numDets + i];
      const h = data[3 * numDets + i];
      let x1 = (cxb - w / 2 - letterbox.padX) / letterbox.scale;
      let y1 = (cyb - h / 2 - letterbox.padY) / letterbox.scale;
      let x2 = (cxb + w / 2 - letterbox.padX) / letterbox.scale;
      let y2 = (cyb + h / 2 - letterbox.padY) / letterbox.scale;
      x1 = Math.max(0, x1);
      y1 = Math.max(0, y1);
      x2 = Math.min(width, x2);
      y2 = Math.min(height, y2);
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
