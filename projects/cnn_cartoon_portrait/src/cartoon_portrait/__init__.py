"""无需预训练模型的自拍人像漫画化管道。"""

from .pipeline import (
    Config,
    Selection,
    compose_rgba,
    extract_lines,
    quantize_colors,
    read_image,
    save_png,
    segment_person,
    smooth_foreground,
)

__all__ = [
    "Config", "Selection", "read_image", "segment_person",
    "smooth_foreground", "quantize_colors", "extract_lines",
    "compose_rgba", "save_png",
]
