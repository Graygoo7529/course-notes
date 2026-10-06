"""可选的桌面标记界面。缩小的是预览，笔划和分割始终使用原图坐标。"""

import cv2
import numpy as np

from .pipeline import FloatImage, Labels, Mask, Selection, segment_person


def collect_selection(image: FloatImage, iterations: int = 5) -> tuple[Mask, Labels]:
    """R 框选，F 前景，B 背景，Enter 计算，Space 接受，Esc 取消。"""
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / width, 800 / height)
    view_w, view_h = max(1, round(width * scale)), max(1, round(height * scale))
    pixels = cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    labels = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    preview_mask: Mask | None = None
    dirty = True
    mode = "r"
    brush = 8  # 显示窗口中的半径；映射回原图后绘制。
    dragging = False
    start = (0, 0)
    previous = (0, 0)
    current = (0, 0)
    message = "R: rectangle | F: foreground | B: background | Enter: calculate"
    window = "Portrait selection"

    def point(x: int, y: int) -> tuple[int, int]:
        return min(width - 1, max(0, round(x * width / view_w))), min(
            height - 1, max(0, round(y * height / view_h)),
        )

    def paint(a: tuple[int, int], b: tuple[int, int]) -> None:
        nonlocal dirty
        value = cv2.GC_FGD if mode == "f" else cv2.GC_BGD
        radius = max(1, round(brush / scale))
        cv2.line(labels, a, b, value, 2 * radius + 1)
        cv2.circle(labels, b, radius, value, -1)
        dirty = True

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        nonlocal dragging, start, previous, current, dirty, message, preview_mask
        if y < 60 and not dragging:
            return
        current = point(x, y - 60)
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            start = previous = current
            if mode != "r":
                paint(current, current)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            if mode != "r":
                paint(previous, current)
            previous = current
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            if mode == "r":
                x0, x1 = sorted((start[0], current[0]))
                y0, y1 = sorted((start[1], current[1]))
                if x1 > x0 and y1 > y0:
                    labels[:] = cv2.GC_BGD
                    labels[y0:y1 + 1, x0:x1 + 1] = cv2.GC_PR_FGD
                    preview_mask = None
                    dirty = True
                    message = "Rectangle ready. Add F/B strokes if needed; Enter calculates."
            else:
                paint(previous, current)

    print("标记窗口：R 框选；F 前景笔；B 背景笔；[ / ] 调笔宽；C 清空。")
    print("Enter 运行分割；补画笔划后再次 Enter；Space 接受；Esc 取消。")
    print("人物接触画面边缘时，可按 C 清空框选，再用 F/B 笔划初始化。")
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)
        while True:
            # 只缩放显示，不对原图、遮罩或处理结果进行上/下采样。
            view = cv2.resize(pixels, (view_w, view_h), interpolation=cv2.INTER_AREA)
            small_labels = cv2.resize(labels, (view_w, view_h), interpolation=cv2.INTER_NEAREST)
            if preview_mask is not None:
                small_mask = cv2.resize(
                    preview_mask.astype(np.uint8), (view_w, view_h), interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                view[~small_mask] = np.rint(view[~small_mask] * 0.25).astype(np.uint8)
            for value, color in ((cv2.GC_FGD, (50, 230, 50)), (cv2.GC_BGD, (230, 90, 30))):
                marked = small_labels == value
                view[marked] = (0.65 * view[marked] + 0.35 * np.array(color)).astype(np.uint8)
            if dragging and mode == "r":
                a = (round(start[0] * view_w / width), round(start[1] * view_h / height))
                b = (round(current[0] * view_w / width), round(current[1] * view_h / height))
                cv2.rectangle(view, a, b, (0, 255, 255), 2)
            canvas = cv2.copyMakeBorder(view, 60, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(
                canvas, f"Mode: {mode.upper()}   Brush: {brush}px   [ / ] size   C clear   Esc cancel",
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
            )
            cv2.putText(canvas, message, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
            # 顶部 60px 是说明栏；鼠标回调坐标去掉该偏移。
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("已取消标记，没有导出结果。")
            if key in (ord("r"), ord("f"), ord("b")):
                mode = chr(key)
            elif key == ord("["):
                brush = max(1, brush - 2)
            elif key == ord("]"):
                brush = min(100, brush + 2)
            elif key == ord("c"):
                labels[:] = cv2.GC_PR_BGD
                preview_mask = None
                dirty = True
                message = "Cleared. Draw F/B strokes, or R rectangle; Enter calculates."
            elif key in (10, 13):
                print("正在原图分辨率上计算 GrabCut……", flush=True)
                try:
                    preview_mask = segment_person(image, Selection(labels=labels, iterations=iterations))
                except (ValueError, cv2.error) as error:
                    print(f"分割未完成：{error}")
                    message = "Need valid foreground AND background. Add F/B strokes; Enter retries."
                else:
                    dirty = False
                    message = "Space: accept | F/B: correct then Enter | R: redraw rectangle"
            elif key == 32:
                if preview_mask is not None and not dirty:
                    return preview_mask, labels.copy()
                message = "Press Enter to calculate the current selection before accepting."
    finally:
        cv2.destroyAllWindows()

