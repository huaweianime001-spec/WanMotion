"""CLIP-based motion transfer for Wan TI2V."""
from .clip_encoder import MotionClipEncoder
from .adapter import MotionContextAdapter
from .model import WanModelWithMotion, wrap_wan_model

__all__ = [
    "MotionClipEncoder",
    "MotionContextAdapter",
    "WanModelWithMotion",
    "wrap_wan_model",
]
