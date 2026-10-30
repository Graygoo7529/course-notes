import { useCallback, useEffect, useRef, useState } from 'react';
import { Aperture, Camera, CameraOff, Download, FlipHorizontal2, Pause, Play, RefreshCw, SlidersHorizontal, X } from 'lucide-react';
import { isTauri, savePng, Runtime } from './runtime';
import { LatestFrameScheduler } from './scheduler';
import type { FrameMeta, Phase, RuntimeInfo, Stats, StyleConfig } from './protocol';

const initial: StyleConfig = { revision: 0, lineDensity: .5, palette: 'natural_warm', mirrored: true };
const emptyStats: Stats = { displayed: 0, dropped: 0, fps: 0, processingMs: 0, latencyMs: 0, faceStatus: '—' };
const labels: Record<Phase, string> = { idle: '相机待机', permission: '等待摄像头权限', warming: '正在预热模型', running: '实时预览', paused: '处理已暂停', capturing: '正在精修', error: '连接中断' };
const blobFrom = (canvas: HTMLCanvasElement, type: string) => new Promise<Blob>((resolve, reject) =>
  canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('无法读取摄像头图像')), type, .9));

export default function App() {
  const video = useRef<HTMLVideoElement>(null), cartoon = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null), runtime = useRef<Runtime | null>(null);
  const queue = useRef<LatestFrameScheduler | null>(null);
  const teardown = useRef<Promise<void>>(Promise.resolve());
  const epoch = useRef(0), session = useRef(''), frame = useRef(0), phaseRef = useRef<Phase>('idle');
  const styleRef = useRef(initial), liveStats = useRef({ ...emptyStats }), ticks = useRef<number[]>([]);
  const [phase, setPhaseState] = useState<Phase>('idle'), [style, setStyle] = useState(initial);
  const [info, setInfo] = useState<RuntimeInfo | null>(null), [message, setMessage] = useState('');
  const [stats, setStats] = useState(emptyStats), [debug, setDebug] = useState(false);
  const [camera, setCamera] = useState<'user' | 'environment'>('user');
  const [backend, setBackend] = useState('openvino'), [mobileView, setMobileView] = useState('cartoon');
  const [still, setStill] = useState<{ blob: Blob; url: string } | null>(null);
  const setPhase = useCallback((value: Phase) => { phaseRef.current = value; setPhaseState(value); }, []);
  const stop = useCallback(async () => {
    ++epoch.current; setPhase('idle');
    const q = queue.current; queue.current = null;
    const service = runtime.current; runtime.current = null;
    const wait = q?.pause();
    stream.current?.getTracks().forEach(track => track.stop()); stream.current = null;
    if (video.current) video.current.srcObject = null;
    const release = async () => {
      try { await service?.close(); } finally { await wait; }
      liveStats.current.fps = 0;
    };
    const closed = teardown.current.then(release, release);
    teardown.current = closed.catch(() => {});
    await closed;
  }, [setPhase]);
  const fail = useCallback((error: Error) => {
    void stop().finally(() => { setMessage(error.message); setPhase('error'); });
  }, [stop, setPhase]);
  useEffect(() => {
    const visibility = () => { if (document.hidden) { void stop(); setMessage('已在后台关闭相机，返回后可重新开启。'); } };
    document.addEventListener('visibilitychange', visibility);
    const timer = setInterval(() => {
      const now = performance.now(); ticks.current = ticks.current.filter(t => now - t < 1000);
      setStats({ ...liveStats.current, fps: ticks.current.length, dropped: queue.current?.dropped ?? liveStats.current.dropped });
    }, 1000);
    return () => { clearInterval(timer); document.removeEventListener('visibilitychange', visibility); void stop(); };
  }, [stop]);
  useEffect(() => () => { if (still) URL.revokeObjectURL(still.url); }, [still]);

  const start = async (facing = camera) => {
    if (['permission', 'warming', 'capturing'].includes(phaseRef.current)) return;
    await stop(); const generation = ++epoch.current;
    setMessage(''); setPhase('permission');
    try {
      if (!navigator.mediaDevices) throw new Error('摄像头需要 HTTPS 或 localhost 安全上下文');
      const media = await navigator.mediaDevices.getUserMedia({ audio: false,
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } } });
      if (generation !== epoch.current) { media.getTracks().forEach(t => t.stop()); return; }
      stream.current = media;
      media.getVideoTracks()[0].onended = () => fail(new Error('摄像头已断开，请重新开启'));
      const v = video.current; if (!v) throw new Error('视频预览不可用');
      v.srcObject = media; await v.play();
      if (generation !== epoch.current) return;
      setPhase('warming'); session.current = crypto.randomUUID(); frame.current = 0;
      const service = new Runtime(); runtime.current = service;
      const ready = await service.connect(session.current, backend);
      if (generation !== epoch.current) { await service.close(); return; }
      setInfo(ready); ticks.current = []; liveStats.current = { ...emptyStats };
      queue.current = new LatestFrameScheduler(session.current, styleRef.current.revision,
        service.process.bind(service), async result => {
          const image = await createImageBitmap(result.image);
          try {
            if (generation !== epoch.current || phaseRef.current !== 'running' ||
                !queue.current?.accepts(result.meta)) return;
            const canvas = cartoon.current; if (!canvas) return;
            canvas.width = image.width; canvas.height = image.height;
            canvas.getContext('2d')?.drawImage(image, 0, 0);
            const now = performance.now(); ticks.current.push(now);
            liveStats.current = { ...liveStats.current, displayed: liveStats.current.displayed + 1,
              processingMs: result.meta.processingMs, latencyMs: now - result.meta.capturedAtMs, faceStatus: result.meta.faceStatus };
          } finally { image.close(); }
        }, fail);
      setPhase('running');
    } catch (error) { if (generation === epoch.current) fail(error as Error); }
  };

  // A video callback only encodes when the worker is idle; no pending pixel queue.
  useEffect(() => {
    if (phase !== 'running') return;
    const v = video.current; if (!v) return;
    let cancelled = false, id = 0, encoding = false, last = 0;
    const canvas = document.createElement('canvas');
    const schedule = () => {
      if (!cancelled) id = 'requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
    };
    const tick = (now: number) => {
      schedule();
      const q = queue.current;
      if (cancelled || phaseRef.current !== 'running' || !q || now - last < 100 || v.readyState < 2 || !v.videoWidth) return;
      if (q.busy || encoding) { q.dropped++; return; }
      last = now; encoding = true;
      const generation = epoch.current, config = { ...styleRef.current };
      const ratio = Math.min(1, 256 / Math.max(v.videoWidth, v.videoHeight));
      canvas.width = Math.max(1, Math.round(v.videoWidth * ratio));
      canvas.height = Math.max(1, Math.round(v.videoHeight * ratio));
      canvas.getContext('2d')!.drawImage(v, 0, 0, canvas.width, canvas.height);
      const meta: FrameMeta = { sessionId: session.current, frameId: ++frame.current,
        configRevision: config.revision, capturedAtMs: performance.now(), width: canvas.width, height: canvas.height,
        pixelFormat: 'jpeg', rotation: 0, kind: 'preview', style: config };
      void blobFrom(canvas, 'image/jpeg').then(blob => {
        if (!cancelled && generation === epoch.current && styleRef.current.revision === config.revision) q.submit(meta, blob);
      }).catch(error => { if (generation === epoch.current) fail(error); }).finally(() => { encoding = false; });
    };
    schedule();
    return () => { cancelled = true; if ('cancelVideoFrameCallback' in v) v.cancelVideoFrameCallback(id); else cancelAnimationFrame(id); };
  }, [phase, fail]);

  const update = (part: Partial<StyleConfig>) => {
    const next = { ...styleRef.current, ...part, revision: styleRef.current.revision + 1 };
    styleRef.current = next; setStyle(next); queue.current?.configure(next.revision);
  };
  const pause = async () => {
    if (phaseRef.current === 'running') { setPhase('paused'); await queue.current?.pause(); }
    else if (phaseRef.current === 'paused') { runtime.current?.reset(); queue.current?.resume(); setPhase('running'); }
  };
  const capture = async () => {
    const v = video.current, service = runtime.current, q = queue.current;
    if (!v?.videoWidth || !service || !q || !['running', 'paused'].includes(phaseRef.current)) return;
    const generation = epoch.current;
    // Freeze the ORIGINAL camera frame immediately, before awaiting current inference.
    const canvas = document.createElement('canvas'); canvas.width = v.videoWidth; canvas.height = v.videoHeight;
    canvas.getContext('2d')!.drawImage(v, 0, 0);
    const config = { ...styleRef.current }, timestamp = performance.now();
    setPhase('capturing');
    try {
      await q.pause();
      const blob = await blobFrom(canvas, 'image/png');
      if (generation !== epoch.current) return;
      const meta: FrameMeta = { sessionId: session.current, frameId: ++frame.current, configRevision: config.revision,
        capturedAtMs: timestamp, width: canvas.width, height: canvas.height, pixelFormat: 'png', rotation: 0, kind: 'capture', style: config };
      const result = await service.process(meta, blob);
      if (generation !== epoch.current) return;
      setStill({ blob: result.image, url: URL.createObjectURL(result.image) });
      setPhase('paused'); setMessage('精修完成。保存方向与拍照时的镜像设置一致。');
    } catch (error) { if (generation === epoch.current) fail(error as Error); }
  };
  const active = ['running', 'paused', 'capturing'].includes(phase);
  const changing = ['permission', 'warming', 'capturing'].includes(phase);
  return <main className="app-shell">
    <header className="topbar"><div className="brand"><Aperture size={34}/><div><p className="eyebrow">FRAME / INK</p><h1>实时漫画相机</h1></div></div><span className={'phase phase-' + phase}><i/>{labels[phase]}</span></header>
    <div className="intro"><h2>让每一帧，留下笔触。</h2><p>原始画面与漫画并排呈现。保持自然光线，静止片刻，拍下一张精修画像。</p></div>
    <div className="mobile-toggle"><button className={mobileView === 'original' ? 'selected' : ''} onClick={() => setMobileView('original')}>原图</button><button className={mobileView === 'cartoon' ? 'selected' : ''} onClick={() => setMobileView('cartoon')}>漫画</button></div>
    <section className={'view-grid show-' + mobileView}>
      <figure className="original-view"><div className="media-frame"><video ref={video} className={style.mirrored ? 'mirrored' : ''} playsInline muted/>{!active && phase !== 'warming' && <div className="empty"><Camera size={36}/><span>开启相机，画面将在这里出现</span></div>}</div><figcaption><span>01 / 原始画面</span><span>完整画幅 · 无中心裁剪</span></figcaption></figure>
      <figure className="cartoon-view"><div className="media-frame cartoon-frame"><canvas ref={cartoon} className={style.mirrored ? 'mirrored' : ''}/>{!active && <div className="empty"><Aperture size={38}/><span>{phase === 'warming' ? '正在加载本机模型…' : '你的漫画，随画面更新'}</span></div>}</div><figcaption><span>02 / 漫画预览</span><span>{stats.fps > 0 ? stats.fps + ' FPS' : '等待处理'} · MobileNetV3</span></figcaption></figure>
    </section>
    <section className="controls"><div className="button-row">
      <button className="primary" disabled={changing} onClick={() => active ? void stop() : void start()}>{active ? <CameraOff/> : <Camera/>}{active ? '停止相机' : '开启相机'}</button>
      <button disabled={!['running','paused'].includes(phase)} onClick={() => void pause()}>{phase === 'paused' ? <Play/> : <Pause/>}{phase === 'paused' ? '继续预览' : '暂停处理'}</button>
      <button className="capture-button" disabled={!['running','paused'].includes(phase)} onClick={() => void capture()}><Aperture/>{phase === 'capturing' ? '正在精修…' : '拍照精修'}</button>
      <button disabled={changing} onClick={() => { const next = camera === 'user' ? 'environment' : 'user'; setCamera(next); if (active) void start(next); }}><RefreshCw/>{camera === 'user' ? '切换后置' : '切换前置'}</button>
      <button className={'icon-button ' + (style.mirrored ? 'selected' : '')} disabled={changing} aria-label="切换镜像" title="镜像显示与保存" onClick={() => update({ mirrored: !style.mirrored })}><FlipHorizontal2/></button>
    </div><div className="option-row"><label className="density">笔画密度 <input aria-label="笔画密度" type="range" min="0" max="1" step=".05" value={style.lineDensity} disabled={changing} onChange={e => update({ lineDensity: +e.target.value })}/><output>{Math.round(style.lineDensity * 100)}%</output></label>
      <div className="palette"><span>色板</span><button className={style.palette === 'natural_warm' ? 'selected' : ''} disabled={changing} onClick={() => update({ palette: 'natural_warm' })}><i className="warm"/>自然暖色</button><button className={style.palette === 'pastel' ? 'selected' : ''} disabled={changing} onClick={() => update({ palette: 'pastel' })}><i className="pastel"/>柔和粉彩</button></div>
      <button className="text-button" onClick={() => setDebug(!debug)}><SlidersHorizontal/>运行信息</button></div></section>
    {message && <p role="status" className={'notice ' + (phase === 'error' ? 'error' : '')}>{message}</p>}
    {debug && <section className="debug">
      <label>推理后端<select aria-label="推理后端" value={backend} disabled={phase !== 'idle' && phase !== 'error'} onChange={e => setBackend(e.target.value)}><option value="openvino">OpenVINO CPU</option><option value="pytorch">PyTorch CPU</option></select></label>
      <Cell title="当前后端 / 尺寸" value={info ? info.backend + ' / 256 px' : '未连接'}/>
      <Cell title="漫画更新率" value={stats.fps + ' FPS'}/><Cell title="算法 / 提交到显示" value={stats.processingMs.toFixed(1) + ' / ' + stats.latencyMs.toFixed(1) + ' ms'}/>
      <Cell title="已显示 / 跳过帧" value={stats.displayed + ' / ' + stats.dropped}/><Cell title="关键点 / 冷启动" value={stats.faceStatus + (info ? ' / ' + info.coldStartMs.toFixed(0) + ' ms' : '')}/>
    </section>}
    <footer><span><i className="local-dot"/>画面在本机处理</span><span>256 实时预览 · 384 样条精修 · {isTauri() ? 'Tauri 桌面' : '本机开发模式'}</span></footer>
    {still && <div className="modal" role="dialog" aria-label="精修结果"><div className="still-card"><button className="close" aria-label="关闭精修结果" onClick={() => setStill(null)}><X/></button><h2>这一帧，已成画。</h2><img src={still.url} alt="当前冻结原帧生成的精修漫画"/><button className="primary" onClick={() => { void savePng(still.blob).then(path => setMessage('已保存：' + path)).catch(fail); }}><Download/>保存 PNG</button><p>已按拍照时的镜像设置处理方向。</p></div></div>}
  </main>;
}
function Cell({ title, value }: { title: string; value: string }) { return <div><b>{title}</b><span>{value}</span></div>; }
