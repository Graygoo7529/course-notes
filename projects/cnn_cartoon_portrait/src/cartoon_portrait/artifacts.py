"""结果导出与合成测试图，不参与七个核心函数的计算。"""

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pipeline import Config, FloatImage, Labels, Mask, save_png


def load_mask(path: Path, shape: tuple[int, ...]) -> Mask:
    with Image.open(path) as opened:
        array = np.asarray(opened)
    if array.ndim != 2 or array.shape != shape[:2]:
        raise ValueError("mask 必须是与已校正方向的输入图片同尺寸的单通道 PNG。")
    if not np.isin(array, [0, 255]).all():
        raise ValueError("mask 使用 0 表示背景、255 表示前景，不接受灰色透明度。")
    mask = array == 255
    if not mask.any():
        raise ValueError("mask 没有前景。")
    return mask


def load_labels(path: Path, shape: tuple[int, ...]) -> Labels:
    with Image.open(path) as opened:
        array = np.asarray(opened)
    if array.dtype != np.uint8 or array.ndim != 2 or array.shape != shape[:2]:
        raise ValueError("labels 必须是与输入图片同尺寸的 8 位单通道 PNG。")
    if np.any(array > 3):
        raise ValueError("labels 像素只能是 GrabCut 的 0、1、2、3 标签。")
    return array.copy()


def _save_rgb(array: FloatImage, path: Path) -> None:
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def preview(rgba: FloatImage, background: float) -> FloatImage:
    """仅用于显示：一次性执行 straight-alpha 合成。"""
    alpha = rgba[..., 3:4]
    return (rgba[..., :3] * alpha + background * (1 - alpha)).astype(np.float32)


def export_results(
    directory: Path,
    image: FloatImage,
    mask: Mask,
    base: FloatImage,
    colors: FloatImage,
    lines: FloatImage,
    rgba: FloatImage,
    config: Config,
    metadata: dict[str, object],
    labels: Labels | None = None,
) -> Path:
    """每次使用新目录，避免覆盖原图或上一次实验。"""
    directory.mkdir(parents=True, exist_ok=False)
    _save_rgb(image, directory / "input.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(directory / "mask.png")
    _save_rgb(base, directory / "base.png")
    _save_rgb(colors, directory / "colors.png")
    # 白底黑线便于肉眼阅读；文件显示值为 1-E，而非函数中的 E。
    Image.fromarray(np.rint((1 - lines) * 255).astype(np.uint8)).save(directory / "lines.png")
    save_png(rgba, directory / "cartoon.png")
    light = preview(rgba, 1.0)
    dark = preview(rgba, 0.12)
    _save_rgb(light, directory / "preview_light.png")
    _save_rgb(dark, directory / "preview_dark.png")
    if labels is not None:
        Image.fromarray(labels).save(directory / "selection_labels.png")
    details = {
        **metadata,
        "config": asdict(config),
        "shape": list(image.shape),
        "foreground_fraction": float(mask.mean()),
        "alpha": "binary, straight (not premultiplied)",
        "lines_png": "display intensity = 1 - E; white means no ink",
        "dependencies": {name: version(name) for name in ("numpy", "opencv-python", "Pillow")},
    }
    (directory / "parameters.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    _overview(directory, image, mask, base, colors, lines, light)
    return directory / "cartoon.png"


def _overview(
    directory: Path, image: FloatImage, mask: Mask, base: FloatImage,
    colors: FloatImage, lines: FloatImage, result: FloatImage,
) -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default(size=19)
    captions = (
        ["1 原图", "2 人物遮罩", "3 卷积平滑", "4 明度量化", "5 Sobel 描线", "6 漫画合成"]
        if font_path.exists()
        else ["1 Input", "2 Foreground mask", "3 Gaussian smoothing", "4 Lab quantization", "5 Sobel lines", "6 Cartoon"]
    )
    board = Image.new("RGB", (1080, 760), "#eeeeee")
    draw = ImageDraw.Draw(board)
    arrays = [
        image,
        np.repeat(mask[..., None].astype(np.float32), 3, axis=2),
        np.where(mask[..., None], base, np.float32(1)),
        np.where(mask[..., None], colors, np.float32(1)),
        np.repeat((1 - lines)[..., None], 3, axis=2),
        result,
    ]
    for index, (caption, array) in enumerate(zip(captions, arrays)):
        left, top = (index % 3) * 360, (index // 3) * 380
        tile = Image.fromarray(np.rint(array * 255).astype(np.uint8))
        tile.thumbnail((336, 326), Image.Resampling.LANCZOS)
        board.paste(tile, (left + (360 - tile.width) // 2, top + 42 + (326 - tile.height) // 2))
        draw.text((left + 14, top + 9), caption, font=font, fill="#222222")
    board.save(directory / "overview.png")


def make_demo() -> tuple[FloatImage, Mask, tuple[int, int, int, int]]:
    """已知前景的合成人像图，用于验证运算；不模拟真实自拍的分割难度。"""
    width, height = 480, 560
    canvas = Image.new("RGB", (width, height), (90, 150, 175))
    draw = ImageDraw.Draw(canvas)
    for y in range(height):
        draw.line((0, y, width, y), fill=(80 + y // 16, 145 + y // 22, 175 + y // 30))
    draw.rectangle((25, 30, 110, 200), fill=(150, 190, 205))
    draw.rectangle((370, 120, 450, 400), fill=(110, 165, 185))
    foreground = Image.new("L", (width, height), 0)
    stencil = ImageDraw.Draw(foreground)
    # 肩部特意接触画面底边：框选允许触底，不能机械地把四边都当背景。
    shapes = [
        ("ellipse", (62, 390, 418, 720), (44, 72, 120)),
        ("rectangle", (202, 330, 278, 441), (213, 152, 111)),
        ("ellipse", (132, 71, 348, 385), (39, 32, 33)),
        ("ellipse", (148, 118, 332, 371), (226, 169, 126)),
        ("ellipse", (133, 62, 340, 198), (39, 32, 33)),
    ]
    for kind, box, color in shapes:
        if kind == "ellipse":
            draw.ellipse(box, fill=color)
            stencil.ellipse(box, fill=255)
        else:
            draw.rectangle(box, fill=color)
            stencil.rectangle(box, fill=255)
    draw.line((173, 221, 211, 216), fill=(59, 38, 31), width=5)
    draw.line((267, 216, 308, 223), fill=(59, 38, 31), width=5)
    draw.ellipse((180, 236, 211, 249), fill=(244, 236, 220))
    draw.ellipse((268, 236, 299, 249), fill=(244, 236, 220))
    draw.ellipse((193, 236, 202, 248), fill=(35, 27, 26))
    draw.ellipse((277, 236, 286, 248), fill=(35, 27, 26))
    draw.line((241, 245, 229, 287, 245, 291), fill=(168, 108, 83), width=3)
    draw.arc((210, 293, 274, 327), 10, 168, fill=(132, 65, 63), width=4)
    image = np.asarray(canvas, dtype=np.float32) / 255
    mask = np.asarray(foreground) == 255
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:height, :width]
    shading = (0.04 * np.sin(xx / 80) + 0.02 * np.cos(yy / 50))[..., None]
    noise = rng.normal(0, 0.009, image.shape)
    image = np.clip(image + (shading + noise) * mask[..., None], 0, 1).astype(np.float32)
    return image, mask, (48, 45, 384, height - 45)

