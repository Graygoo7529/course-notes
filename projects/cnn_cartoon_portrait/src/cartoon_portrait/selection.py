"""显示和鼠标事件层；图像计算、笔划标签始终保留原图坐标。"""

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import (
    FloatImage, Labels, Mask, Selection, _integer, _mask, _number, _unit_array, segment_person,
)


@dataclass(frozen=True)
class ViewConfig:
    max_scale: float = 4.0
    max_width: int = 1200
    max_height: int = 800
    min_canvas_width: int = 760
    toolbar_height: int = 110
    brush_radius: int = 2

    def validate(self) -> None:
        _number(self.max_scale, "max_scale", float(np.finfo(np.float32).eps))
        for name in ("max_width", "max_height", "min_canvas_width", "brush_radius"):
            _integer(getattr(self, name), name, 1)
        _integer(self.min_canvas_width, "min_canvas_width", 600)
        _integer(self.toolbar_height, "toolbar_height", 100)


@dataclass(frozen=True)
class ViewTransform:
    image_width: int
    image_height: int
    view_width: int
    view_height: int
    offset_x: int
    offset_y: int

    @property
    def scale_x(self) -> float:
        return self.view_width / self.image_width

    @property
    def scale_y(self) -> float:
        return self.view_height / self.image_height


def view_to_image(
    transform: ViewTransform, x: int, y: int, *, clip: bool = False,
) -> tuple[int, int] | None:
    """显示像素中心→最近原图像素；留白不接受新的标记。"""
    u, v = x - transform.offset_x, y - transform.offset_y
    if not clip and not (0 <= u < transform.view_width and 0 <= v < transform.view_height):
        return None
    column = int(np.floor((u + 0.5) / transform.scale_x))
    row = int(np.floor((v + 0.5) / transform.scale_y))
    return (
        min(transform.image_width - 1, max(0, column)),
        min(transform.image_height - 1, max(0, row)),
    )


def view_rect_to_image(
    transform: ViewTransform, start: tuple[int, int], end: tuple[int, int],
) -> tuple[int, int, int, int]:
    """矩形按像素覆盖区域转换：左上 floor，右下 ceil，返回半开区间。"""
    left, right = sorted((start[0] - transform.offset_x, end[0] - transform.offset_x))
    top, bottom = sorted((start[1] - transform.offset_y, end[1] - transform.offset_y))
    x0 = int(np.clip(np.floor(left / transform.scale_x), 0, transform.image_width))
    y0 = int(np.clip(np.floor(top / transform.scale_y), 0, transform.image_height))
    x1 = int(np.clip(np.ceil((right + 1) / transform.scale_x), 0, transform.image_width))
    y1 = int(np.clip(np.ceil((bottom + 1) / transform.scale_y), 0, transform.image_height))
    return x0, y0, x1 - x0, y1 - y0


def _wrapped(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def make_preview(
    image: FloatImage, labels: Labels, mask: Mask | None = None,
    config: ViewConfig = ViewConfig(), *, mode: str = "r", brush: int = 2,
    message: str = "R: rectangle | F/B: strokes | Enter: calculate | Space: accept",
) -> tuple[Labels, ViewTransform]:
    """纯函数：返回 BGR 显示画布与实际坐标变换，不打开窗口。"""
    config.validate()
    _unit_array(image, "image", 3)
    height, width = image.shape[:2]
    if labels.shape != (height, width) or labels.dtype != np.uint8 or np.any(labels > 3):
        raise ValueError("labels 必须是原图尺寸的 0/1/2/3 标签。")
    scale = min(config.max_scale, config.max_width / width, config.max_height / height)
    view_w, view_h = max(1, round(width * scale)), max(1, round(height * scale))
    canvas_w = max(config.min_canvas_width, view_w)
    transform = ViewTransform(width, height, view_w, view_h, (canvas_w - view_w) // 2, config.toolbar_height)
    pixels = cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    view = cv2.resize(
        pixels, (view_w, view_h),
        interpolation=cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA,
    )
    resized_labels = cv2.resize(labels, (view_w, view_h), interpolation=cv2.INTER_NEAREST_EXACT)
    if mask is not None:
        _mask(mask, image.shape)
        resized_mask = cv2.resize(
            mask.astype(np.uint8), (view_w, view_h), interpolation=cv2.INTER_NEAREST_EXACT,
        ).astype(bool)
        view[~resized_mask] = np.rint(view[~resized_mask] * .25).astype(np.uint8)
    for value, color in ((cv2.GC_FGD, (50, 230, 50)), (cv2.GC_BGD, (230, 90, 30))):
        marked = resized_labels == value
        view[marked] = (.65 * view[marked] + .35 * np.asarray(color)).astype(np.uint8)
    canvas = np.full((config.toolbar_height + view_h, canvas_w, 3), 25, dtype=np.uint8)
    canvas[transform.offset_y:, transform.offset_x:transform.offset_x + view_w] = view
    instructions = [
        f"Mode: {mode.upper()} | Brush: {brush}px in original | [ / ] size | C clear | Esc cancel",
        f"View: {view_w} x {view_h} | Original: {width} x {height}",
        message,
    ]
    lines = [part for instruction in instructions for part in _wrapped(instruction, canvas_w - 20)]
    for index, text in enumerate(lines[:5]):
        cv2.putText(
            canvas, text, (10, 18 + index * 20), cv2.FONT_HERSHEY_SIMPLEX,
            .45, (240, 240, 240), 1,
        )
    return canvas, transform


def collect_selection(
    image: FloatImage, iterations: int = 5, view_config: ViewConfig = ViewConfig(),
) -> tuple[Mask, Labels]:
    """R 框选，F/B 原图像素笔划，Enter 分割，Space 接受；窗口状态限于本层。"""
    view_config.validate()
    labels = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    preview_mask: Mask | None = None
    dirty, dragging = True, False
    mode, brush = "r", view_config.brush_radius
    start = current = (0, 0)
    previous: tuple[int, int] | None = None
    cursor: tuple[int, int] | None = None
    message = "R: rectangle | F/B: strokes | Enter: calculate | Space: accept"
    _, transform = make_preview(image, labels, config=view_config)
    window = "Portrait selection"

    def paint(a: tuple[int, int], b: tuple[int, int]) -> None:
        nonlocal dirty
        value = cv2.GC_FGD if mode == "f" else cv2.GC_BGD
        cv2.line(labels, a, b, value, 2 * brush + 1)
        cv2.circle(labels, b, brush, value, -1)
        dirty = True

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        nonlocal dragging, start, current, previous, cursor, dirty, message, preview_mask
        point = view_to_image(transform, x, y)
        cursor = (x, y) if point is not None else None
        if event == cv2.EVENT_LBUTTONDOWN and point is not None:
            dragging, start, current, previous = True, (x, y), (x, y), point
            if mode != "r":
                paint(point, point)
        elif event == cv2.EVENT_MOUSEMOVE and dragging and point is not None:
            current = (x, y)
            if mode != "r" and previous is not None:
                paint(previous, point)
            previous = point
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            current = (x, y)
            if mode == "r":
                x0, y0, w, h = view_rect_to_image(transform, start, current)
                if w > 0 and h > 0:
                    labels[:] = cv2.GC_BGD
                    labels[y0:y0 + h, x0:x0 + w] = cv2.GC_PR_FGD
                    preview_mask, dirty = None, True
                    message = "Rectangle ready. Add F/B strokes; Enter calculates."
            elif point is not None and previous is not None:
                paint(previous, point)

    print("标记窗口：R 框选；F 前景笔；B 背景笔；[ / ] 调原图笔宽；C 清空。")
    print("Enter 运行分割；补画后再次 Enter；Space 接受；Esc 取消。")
    print(f"预览尺寸：{transform.view_width} × {transform.view_height}；仅放大显示。")
    print("人物接触边缘时，可 C 清空框选，再用 F/B 笔划初始化。")
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)
        while True:
            canvas, transform = make_preview(
                image, labels, preview_mask, view_config, mode=mode, brush=brush, message=message,
            )
            if dragging and mode == "r":
                cv2.rectangle(canvas, start, current, (0, 255, 255), 1)
            elif cursor is not None and mode != "r":
                axes = (max(1, round(brush * transform.scale_x)), max(1, round(brush * transform.scale_y)))
                cv2.ellipse(canvas, cursor, axes, 0, 0, 360, (0, 255, 255), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("已取消标记，没有导出结果。")
            if key in (ord("r"), ord("f"), ord("b")):
                mode, dragging = chr(key), False
            elif key == ord("["):
                brush = max(1, brush - 1)
            elif key == ord("]"):
                brush = min(100, brush + 1)
            elif key == ord("c"):
                labels[:] = cv2.GC_PR_BGD
                preview_mask, dirty, dragging = None, True, False
                message = "Cleared. Draw F/B strokes or R rectangle; Enter calculates."
            elif key in (10, 13):
                print("正在原图分辨率上计算 GrabCut……", flush=True)
                try:
                    preview_mask = segment_person(image, Selection(labels=labels, iterations=iterations))
                except (ValueError, cv2.error) as error:
                    dirty = True
                    print(f"分割未完成：{error}")
                    message = "Need foreground AND background. Add F/B strokes; Enter retries."
                else:
                    dirty = False
                    message = "Space: accept | F/B: correct then Enter | R: redraw rectangle"
            elif key == 32:
                if preview_mask is not None and not dirty:
                    return preview_mask, labels.copy()
                message = "Press Enter to calculate the current selection before accepting."
    finally:
        cv2.destroyWindow(window)
