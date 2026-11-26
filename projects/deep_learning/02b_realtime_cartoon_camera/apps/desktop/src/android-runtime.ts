import { invoke } from '@tauri-apps/api/core';
import type { FrameMeta, FrameResult, RuntimeInfo } from './protocol';

export const toBase64 = (blob: Blob) => new Promise<string>((resolve, reject) => {
  const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]);
  reader.onerror = () => reject(reader.error); reader.readAsDataURL(blob);
});

export class AndroidRuntime {
  private sessionId = '';
  private initializing: Promise<RuntimeInfo> | null = null;
  private closed = false;
  async connect(sessionId: string): Promise<RuntimeInfo> {
    this.sessionId = sessionId;
    this.initializing = invoke('plugin:cartoon|initialize', { payload: { sessionId } });
    const result = await this.initializing;
    if (this.closed) throw new Error('Session stopped');
    return result;
  }
  async process(meta: FrameMeta, image: Blob): Promise<FrameResult> {
    if (this.closed) throw new Error('Session stopped');
    const result = await invoke<{ meta: FrameResult['meta']; image: string }>('plugin:cartoon|process_frame', {
      payload: { sessionId: this.sessionId, meta: JSON.stringify(meta), image: await toBase64(image) },
    });
    return { meta: result.meta, image: new Blob([Uint8Array.from(atob(result.image), c => c.charCodeAt(0))], { type: result.meta.mime }) };
  }
  reset() { void invoke('plugin:cartoon|reset', { payload: { sessionId: this.sessionId } }).catch(() => {}); }
  async close() {
    this.closed = true;
    await this.initializing?.catch(() => null);
    await invoke('plugin:cartoon|stop', { payload: { sessionId: this.sessionId } });
  }
}
