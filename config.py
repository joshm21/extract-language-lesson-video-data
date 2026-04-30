from functools import partial

from core import timestamps
from core import extract
from core import prepare
from core import mask
from core import detect
from core import score
from core import filter as filt
from core import crop
from core import dedupe

CONFIG = {
    # TIME: Video -> List of Timestamps
    "timer": partial(timestamps.uniform, count=6),

    # EXTRACT: Video + List of Timestamps -> Raw Image Frames
    "extractor": extract.extract_frame_at_time,

    # PROCESS: Raw Image Frame -> Processed Image
    "processing": [
        {"name": "gray",    "func": partial(prepare.get_grayscale)},
        {"name": "blurred", "func": partial(prepare.apply_blur, ksize=7)},
        {"name": "canny",   "func": partial(mask.canny, low=50, high=150)},
        {"name": "closed",  "func": partial(
            prepare.apply_close, kernel_size=3)},
    ],

    # DETECT: Find the initial quads
    "detector": partial(detect.get_quads, min_area=300),

    # SCORE: Score each quad
    "scorer": partial(score.score_quad),

    # FILTER: Quad -> Boolean (Keep/Drop)
    "filters": [
        {"name": "area",   "func": lambda q: q.score.relative_area > 0.001},
        {"name": "convex", "func": lambda q: q.score.convexity > 0.9},
        {"name": "sat",    "func": lambda q: q.score.saturation > 20},
    ],

    # CROP: Raw Image Frame + Quad -> Cropped Card
    "cropper": crop.all,

    # DEDUPE: Get unique cropped cards
    "deduper": partial(dedupe.get_unique, threshold=10)
}
