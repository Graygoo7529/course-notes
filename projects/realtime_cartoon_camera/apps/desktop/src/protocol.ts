export type Phase = 'idle' | 'permission' | 'warming' | 'running' | 'paused' | 'capturing' | 'error';
export interface StyleConfig { revision: number; lineDensity: number; palette: 'natural_warm' | 'pastel'; mirrored: boolean; }
export interface FrameMeta {
  sessionId: string; frameId: number; configRevision: number; capturedAtMs: number;
  width: number; height: number; pixelFormat: 'jpeg' | 'png'; rotation: 0; kind: 'preview' | 'capture'; style: StyleConfig;
}
export interface RuntimeInfo { backend: string; model: string; inputSize: number; device: string; coldStartMs: number; }
export interface FrameResult { meta: FrameMeta & { processingMs: number; mime: string; faceStatus: string; renderer: string }; image: Blob; }
export interface Stats { displayed: number; dropped: number; fps: number; processingMs: number; latencyMs: number; faceStatus: string; }
