"""
roop/globals.py
Mask settings are read from environment variables set by handler.py.
This means handler.py controls them without any CLI flag changes.
"""
import os
from typing import List, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None

# ── Processing options ────────────────────────────────────────────────────────
frame_processors: List[str] = []
keep_fps:    Optional[bool] = None
keep_frames: Optional[bool] = None
skip_audio:  Optional[bool] = None
many_faces:  Optional[bool] = None
reference_face_position: int   = 0
reference_frame_number:  int   = 0
similar_face_distance:   float = 0.85

# ── Execution ─────────────────────────────────────────────────────────────────
headless:            Optional[bool] = None
log_level:           str            = 'error'
execution_providers: List[str]      = ['CPUExecutionProvider']
execution_threads:   int            = 4
max_memory:          Optional[int]  = None

# ── Output quality ────────────────────────────────────────────────────────────
temp_frame_format:    str = 'png'
temp_frame_quality:   int = 0
output_video_encoder: str = 'libx264'
output_video_quality: int = 35

# ── Face mask settings — read from env vars set by handler.py ─────────────────
# handler.py sets: ROOP_FACE_MASK_TYPE, ROOP_FACE_MASK_BLUR,
#                  ROOP_FACE_MASK_PADDING, ROOP_FACE_MASK_REGIONS,
#                  ROOP_USE_YOLO_FACE_DETECTOR

face_mask_type: str = os.environ.get('ROOP_FACE_MASK_TYPE', 'box')

face_mask_blur: float = float(os.environ.get('ROOP_FACE_MASK_BLUR', '0.3'))

# ROOP_FACE_MASK_PADDING is "top,right,bottom,left" e.g. "5,5,5,5"
_pad_str = os.environ.get('ROOP_FACE_MASK_PADDING', '0,0,0,0')
try:
    _pad_parts = [int(x) for x in _pad_str.split(',')]
    face_mask_padding: tuple = tuple(_pad_parts) if len(_pad_parts) == 4 else (0, 0, 0, 0)
except (ValueError, AttributeError):
    face_mask_padding = (0, 0, 0, 0)

# ROOP_FACE_MASK_REGIONS is comma-separated e.g. "skin,nose,mouth"
_regions_str = os.environ.get('ROOP_FACE_MASK_REGIONS', '')
face_mask_regions: List[str] = (
    [r.strip() for r in _regions_str.split(',') if r.strip()]
    if _regions_str
    else ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye',
          'nose', 'mouth', 'upper-lip', 'lower-lip']
)

# ROOP_USE_YOLO_FACE_DETECTOR is "1" or "0"
use_yolo_face_detector: bool = os.environ.get('ROOP_USE_YOLO_FACE_DETECTOR', '0') == '1'
