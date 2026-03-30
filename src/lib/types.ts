export type DetectionMode = 'rectangle' | 'anchor' | 'polygon' | 'exclusive';

export interface SeatConfig {
  id: string;
  box?: [number, number, number, number];
  polygon?: [number, number][];
}

export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

export interface AppConfig {
  mode: DetectionMode;
  seats: SeatConfig[];
}

export type AppPhase = 'configure' | 'setup' | 'detect';

export const VALID_MODES: DetectionMode[] = ['rectangle', 'anchor', 'polygon', 'exclusive'];

export const MODE_INFO: Record<DetectionMode, { label: string; desc: string; icon: string }> = {
  rectangle: {
    label: 'Rectangle',
    desc: 'IoU/overlap method — best for eye-level cameras',
    icon: '▭',
  },
  anchor: {
    label: 'Anchor',
    desc: 'Bottom-center point — best for CCTV / elevated cameras',
    icon: '⊕',
  },
  polygon: {
    label: 'Polygon',
    desc: '4-point trapezoid zones — most accurate for permanent CCTV',
    icon: '⬠',
  },
  exclusive: {
    label: 'Exclusive',
    desc: 'Rectangle overlap, one person = one seat only',
    icon: '⊡',
  },
};
