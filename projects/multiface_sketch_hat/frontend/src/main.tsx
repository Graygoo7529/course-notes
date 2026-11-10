import { StrictMode, useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

type Detection = {
  trackId: number | null;
  score: number;
  bbox: [number, number, number, number];
};

type SceneHeader = {
  kind: 'scene' | 'capture';
  frameId: number;
  width: number;
  height: number;
  detections: Detection[];
  cards: string[];
};

const API_HOST = '127.0.0.1:8000';

function readBinaryResponse(buffer: ArrayBuffer): { header: SceneHeader; jpeg: Blob } {
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, false);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength))) as SceneHeader;
  return { header, jpeg: new Blob([buffer.slice(4 + headerLength)], { type: 'image/jpeg' }) };
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inFlightRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('\u70b9\u51fb\u201c\u6253\u5f00\u6444\u50cf\u5934\u201d\u5f00\u59cb');
  const [sceneUrl, setSceneUrl] = useState<string | null>(null);
  const [cards, setCards] = useState<string[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [lastFrame, setLastFrame] = useState<number | null>(null);

  const closeCamera = useCallback(() => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setSceneUrl((old) => { if (old) URL.revokeObjectURL(old); return null; });
    setCards([]);
    setDetections([]);
    setLastFrame(null);
    inFlightRef.current = false;
    setRunning(false);
    setStatus('\u6444\u50cf\u5934\u5df2\u5173\u95ed');
  }, []);

  useEffect(() => closeCamera, [closeCamera]);

  const sendFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const socket = socketRef.current;
    if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || inFlightRef.current || video.readyState < 2) return;
    const width = Math.min(video.videoWidth, 640);
    const height = Math.round((video.videoHeight / video.videoWidth) * width);
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d')?.drawImage(video, 0, 0, width, height);
    inFlightRef.current = true;
    canvas.toBlob((blob) => {
      if (blob && socket.readyState === WebSocket.OPEN) socket.send(blob);
      else inFlightRef.current = false;
    }, 'image/jpeg', 0.78);
  }, []);

  const openCamera = useCallback(async () => {
    try {
      setStatus('\u6b63\u5728\u8fde\u63a5\u540e\u7aef\u670d\u52a1');
      const socket = new WebSocket(`ws://${API_HOST}/ws/process`);
      socket.binaryType = 'arraybuffer';
      socket.onmessage = (event) => {
        if (typeof event.data === 'string') {
          const message = JSON.parse(event.data) as { kind: string; message?: string };
          setStatus(message.message ?? '\u540e\u7aef\u8fd4\u56de\u9519\u8bef');
          inFlightRef.current = false;
          return;
        }
        const { header, jpeg } = readBinaryResponse(event.data as ArrayBuffer);
        const nextUrl = URL.createObjectURL(jpeg);
        setSceneUrl((old) => { if (old) URL.revokeObjectURL(old); return nextUrl; });
        setDetections(header.detections);
        setLastFrame(header.frameId);
        if (header.kind === 'capture') {
          setCards(header.cards.map((item) => `data:image/jpeg;base64,${item}`));
          setStatus(`\u5df2\u4ece\u7b2c ${header.frameId} \u5e27\u751f\u6210 ${header.cards.length} \u5f20\u9010\u4eba\u5361\u7247`);
        } else {
          setStatus(`\u5b9e\u65f6\u5904\u7406\u4e2d \u00b7 \u7b2c ${header.frameId} \u5e27 \u00b7 ${header.detections.length} \u5f20\u8138`);
        }
        inFlightRef.current = false;
      };
      socket.onclose = () => setRunning(false);
      await new Promise<void>((resolve, reject) => {
        socket.onopen = () => resolve();
        socket.onerror = () => reject(new Error('websocket'));
      });
      socket.onerror = () => {
        setStatus('\u65e0\u6cd5\u8fde\u63a5\u540e\u7aef\uff0c\u8bf7\u786e\u8ba4 FastAPI \u670d\u52a1\u5df2\u542f\u52a8');
        inFlightRef.current = false;
      };
      socketRef.current = socket;
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      streamRef.current = stream;
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setRunning(true);
      setStatus('\u5b9e\u65f6\u5904\u7406\u4e2d');
      timerRef.current = window.setInterval(() => void sendFrame(), 450);
      void sendFrame();
    } catch (error) {
      const message = error instanceof DOMException && error.name === 'NotAllowedError'
        ? '\u6d4f\u89c8\u5668\u62d2\u7edd\u4e86\u6444\u50cf\u5934\u6743\u9650'
        : '\u542f\u52a8\u5931\u8d25\uff0c\u8bf7\u786e\u8ba4\u6444\u50cf\u5934\u548c\u540e\u7aef\u670d\u52a1';
      closeCamera();
      setStatus(message);
    }
  }, [closeCamera, sendFrame]);

  const capture = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN && lastFrame !== null) {
      setStatus('\u6b63\u5728\u5904\u7406\u6700\u540e\u4e00\u5e27\u5e76\u751f\u6210\u9010\u4eba\u5361\u7247');
      socketRef.current.send(JSON.stringify({ kind: 'capture' }));
    }
  };

  return (
    <main className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">{'\u6df1\u5ea6\u5b66\u4e60\u5e94\u7528'} {'\u00b7'} {'\u7b2c\u4e8c\u9636\u6bb5'}</p>
          <h1>{'\u591a\u4eba\u7d20\u63cf\u4e0e\u865a\u62df\u6234\u5e3d'}</h1>
          <p className="lede">{'\u6574\u5e27\u591a\u4eba\u5b9e\u65f6\u9884\u89c8\uff0c\u6309\u4e0b\u62cd\u7167\u540e\u751f\u6210\u9010\u4eba\u8096\u50cf\u5361\u7247\u3002'}</p>
        </div>
        <div className="controls">
          <button onClick={() => void openCamera()} disabled={running}>{'\u6253\u5f00\u6444\u50cf\u5934'}</button>
          <button className="secondary" onClick={capture} disabled={!running || lastFrame === null}>{'\u62cd\u6444\u9010\u4eba\u5361\u7247'}</button>
          <button className="quiet" onClick={closeCamera} disabled={!running}>{'\u5173\u95ed'}</button>
        </div>
      </header>
      <p className="status" role="status">{status}</p>
      <section className="workspace">
        <article className="panel camera-panel">
          <div className="panel-title"><span>{'\u6444\u50cf\u5934\u539f\u56fe'}</span><small>{'\u6d4f\u89c8\u5668\u672c\u5730\u91c7\u96c6'}</small></div>
          <video ref={videoRef} muted playsInline className="camera" />
          <canvas ref={canvasRef} hidden />
        </article>
        <article className="panel scene-panel">
          <div className="panel-title"><span>{'\u7d20\u63cf\u4e0e\u6de1\u5f69\u5e3d\u5b50\u573a\u666f'}</span><small>{detections.length} faces</small></div>
          {sceneUrl ? <img className="scene" src={sceneUrl} alt={'\u591a\u4eba\u7d20\u63cf\u4e0e\u6de1\u5f69\u5e3d\u5b50\u573a\u666f'} /> : <div className="placeholder">{'\u6253\u5f00\u6444\u50cf\u5934\u540e\u663e\u793a\u5b9e\u65f6\u7ed3\u679c'}</div>}
        </article>
      </section>
      <section className="cards-section">
        <div className="panel-title"><span>{'\u9010\u4eba\u8096\u50cf\u5361\u7247'}</span><small>{'\u540c\u4e00\u51bb\u7ed3\u5e27'}</small></div>
        {cards.length === 0 ? <p className="empty">{'\u70b9\u51fb\u201c\u62cd\u6444\u9010\u4eba\u5361\u7247\u201d\uff0c\u751f\u6210\u5f53\u524d\u753b\u9762\u4e2d\u7684\u4eba\u7269\u7d20\u63cf\u3002'} </p> : <div className="cards">{cards.map((card, index) => <img key={`${card.slice(-24)}-${index}`} src={card} alt={`${'\u4eba\u7269\u7d20\u63cf'} ${index + 1}`} />)}</div>}
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
