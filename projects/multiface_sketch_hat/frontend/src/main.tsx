import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const root = document.getElementById('root');
if (!root) throw new Error('Missing root element');

createRoot(root).render(
  <StrictMode>
    <main>
      <p className="label">深度学习应用 · 新作业</p>
      <h1>多人素描与虚拟戴帽</h1>
      <p>从同一帧画面中，为每个人生成素描与戴帽肖像。</p>
      <div className="status" role="status">
        项目已初始化，正在准备模型。摄像头与生成服务尚未接入。
      </div>
    </main>
  </StrictMode>,
);
