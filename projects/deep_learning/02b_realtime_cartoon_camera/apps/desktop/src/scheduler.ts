import type { FrameMeta, FrameResult } from './protocol';

/** Busy camera callbacks skip sampling; next idle callback samples the current frame. */
export class LatestFrameScheduler {
  private active = true;
  private generation = 0;
  private job: Promise<void> | null = null;
  private sessionId: string;
  private revision: number;
  private newest = -1;
  private process: (meta: FrameMeta, image: Blob) => Promise<FrameResult>;
  private display: (result: FrameResult) => Promise<void>;
  private error: (error: Error) => void;
  dropped = 0;
  constructor(session: string, revision: number, process: LatestFrameScheduler['process'], display: LatestFrameScheduler['display'], error: LatestFrameScheduler['error']) {
    this.sessionId = session; this.revision = revision; this.process = process; this.display = display; this.error = error;
  }
  get busy() { return this.job !== null; }
  get enabled() { return this.active; }
  accepts(meta: FrameMeta) { return this.active && meta.sessionId === this.sessionId && meta.configRevision === this.revision && meta.frameId > this.newest; }
  configure(revision: number) { this.revision = revision; this.generation++; }
  async pause() { this.active = false; this.generation++; await this.job; }
  resume() { this.active = true; }
  submit(meta: FrameMeta, image: Blob) {
    if (!this.active) return;
    if (this.job) { this.dropped++; return; }
    const generation = this.generation;
    this.job = this.process(meta, image).then(async result => {
      if (generation === this.generation && this.accepts(result.meta)) {
        await this.display(result); this.newest = result.meta.frameId;
      } else this.dropped++;
    }).catch(error => { if (this.active && generation === this.generation) this.error(error); })
      .finally(() => { this.job = null; });
  }
}
