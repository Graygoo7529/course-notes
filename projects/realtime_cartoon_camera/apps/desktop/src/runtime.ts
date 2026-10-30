import { invoke } from '@tauri-apps/api/core';
import type { FrameMeta, FrameResult, RuntimeInfo } from './protocol';

export const isTauri = () => '__TAURI_INTERNALS__' in window;
export class Runtime {
  private socket: WebSocket | null = null;
  private closed = false;
  private workerToken: string | null = null;
  private descriptorTask: Promise<{ url: string; token: string }> | null = null;
  private pending: { resolve: (result: FrameResult) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> } | null = null;
  async connect(sessionId: string, backend: string): Promise<RuntimeInfo> {
    this.descriptorTask = isTauri() ? invoke('start_worker') :
      fetch('/api/runtime', { cache: 'no-store' }).then(async response => {
        if (!response.ok) throw new Error(await response.text()); return response.json();
      });
    const descriptor = await this.descriptorTask;
    this.workerToken = descriptor.token;
    if (this.closed) {
      throw new Error('Session stopped');
    }
    const socket = new WebSocket(descriptor.url); this.socket = socket; socket.binaryType = 'arraybuffer';
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { socket.close(); reject(new Error('模型初始化超时')); }, 60000);
      socket.onopen = () => socket.send(JSON.stringify({ token: descriptor.token, backend, sessionId }));
      socket.onerror = () => { clearTimeout(timer); reject(new Error('本机算法连接失败')); };
      socket.onclose = () => { clearTimeout(timer); reject(new Error('算法连接已关闭')); this.fail(new Error('算法连接已关闭')); };
      socket.onmessage = event => {
        if (typeof event.data === 'string') {
          const message = JSON.parse(event.data);
          if (message.type === 'ready') { clearTimeout(timer); resolve(message); }
          else { clearTimeout(timer); const error = new Error(message.error ?? '算法错误'); reject(error); this.fail(error); }
          return;
        }
        try {
          const buffer = event.data as ArrayBuffer;
          const size = new DataView(buffer).getUint32(0);
          if (size > 8192 || size + 4 >= buffer.byteLength) throw new Error('无效算法响应');
          const meta = JSON.parse(new TextDecoder().decode(buffer.slice(4, size + 4)));
          const result = { meta, image: new Blob([buffer.slice(size + 4)], { type: meta.mime }) };
          const current = this.pending; this.pending = null;
          if (current) { clearTimeout(current.timer); current.resolve(result); }
        } catch (error) { this.fail(error as Error); }
      };
    });
  }
  private fail(error: Error) {
    if (this.pending) { clearTimeout(this.pending.timer); this.pending.reject(error); this.pending = null; }
  }
  async process(meta: FrameMeta, image: Blob): Promise<FrameResult> {
    if (this.pending) throw new Error('Only one frame may be in flight');
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) throw new Error('算法尚未连接');
    const header = new TextEncoder().encode(JSON.stringify(meta));
    const length = new Uint8Array(4); new DataView(length.buffer).setUint32(0, header.length);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.fail(new Error('帧处理超时')); this.socket?.close(); }, 30000);
      this.pending = { resolve, reject, timer };
      this.socket!.send(new Blob([length, header, image]));
    });
  }
  reset() { if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: 'reset' })); }
  async close() {
    this.closed = true;
    this.fail(new Error('Session stopped')); this.socket?.close(); this.socket = null;
    // Finish a pending launch before another session can reuse the global worker.
    await this.descriptorTask?.catch(() => null);
    if (isTauri() && this.workerToken) await invoke('stop_worker', { token: this.workerToken });
  }
}

export async function savePng(image: Blob): Promise<string> {
  const filename = `cartoon-${Date.now()}.png`;
  if (isTauri()) return invoke('save_png', { bytes: Array.from(new Uint8Array(await image.arrayBuffer())), filename });
  const url = URL.createObjectURL(image); const link = document.createElement('a');
  link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 30000);
  return filename;
}
