"""
roop/globals.py  —  Enhanced with face-mask & YOLO-detector globals
Drop this file into:  roop/globals.py
"""

from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None

# ---------------------------------------------------------------------------
# Processing options
# ---------------------------------------------------------------------------
frame_processors: List[str] = []
keep_fps: Optional[bool] = None
keep_frames: Optional[bool] = None
skip_audio: Optional[bool] = None
many_faces: Optional[bool] = None
reference_face_position: int = 0
reference_frame_number: int = 0
similar_face_distance: float = 0.85

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
headless: Optional[bool] = None
log_level: str = "error"
execution_providers: List[str] = ["CPUExecutionProvider"]
execution_threads: int = 4
max_memory: Optional[int] = None

# ---------------------------------------------------------------------------
# Temp / output quality
# ---------------------------------------------------------------------------
temp_frame_format: str = "png"
temp_frame_quality: int = 0
output_video_encoder: str = "libx264"
output_video_quality: int = 35

# ---------------------------------------------------------------------------
# NEW — Face mask settings
# ---------------------------------------------------------------------------
# Which masking strategy to apply after the inswapper runs.
#   "box"       – rectangle over bbox, blurred at edges  (original roop behaviour)
#   "occlusion" – GrabCut foreground mask; glasses/hands are NOT overwritten
#   "region"    – BiSeNet parser; only selected facial regions are swapped
face_mask_type: str = "box"          # "box" | "occlusion" | "region"

# Blur applied to the edges of the box mask (0.0 = sharp, 1.0 = very soft)
face_mask_blur: float = 0.3          # range 0.0 – 1.0

# Extra pixels to expand the box mask outward (top, right, bottom, left)
face_mask_padding: tuple = (0, 0, 0, 0)

# Facial sub-regions included when face_mask_type == "region"
# Remove entries you do NOT want swapped (e.g. remove "glasses" to keep glasses)
face_mask_regions: List[str] = [
    "skin",
    "left-eyebrow",
    "right-eyebrow",
    "left-eye",
    "right-eye",
    "nose",
    "mouth",
    "upper-lip",
    "lower-lip",
]

# ---------------------------------------------------------------------------
# NEW — YOLO face detector (optional fast pre-filter)
# ---------------------------------------------------------------------------
# Set True to use YOLOv8-face as a quick first pass before InsightFace.
# Requires:  pip install ultralytics
#            models/yolov8n-face.pt  (auto-downloaded by handler)
use_yolo_face_detector: bool = False
