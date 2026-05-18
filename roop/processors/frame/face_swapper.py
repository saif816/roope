"""
Enhanced Roop Face Swapper
--------------------------
Adds on top of the original roop face_swapper.py:
  • YOLO-based face detector (ultralytics YOLOv8-face) as an optional fast detector
  • Three face-mask types that prevent occluding objects from ruining the swap:
      - box      : simple bounding-rectangle mask (default, same as original roop)
      - occlusion: GrabCut-based mask that carves out foreground objects
                   (glasses, hands, microphones …) so they are NOT overwritten
      - region   : landmark-guided mask that swaps only specific facial sub-regions
                   (skin, left-eye, right-eye, nose, mouth, upper-lip, lower-lip …)
  • Configurable blur & padding for the box mask
  • Thread-safe singleton helpers identical to the original module

Drop this file in  roop/processors/frame/face_swapper.py  to replace the original.
All public function signatures are kept identical so the rest of roop needs no changes.

New roop.globals attributes (optional — all have sensible defaults):
  face_mask_type        : str   = "box"          # "box" | "occlusion" | "region"
  face_mask_blur        : float = 0.3            # 0.0 – 1.0
  face_mask_padding     : tuple = (0, 0, 0, 0)  # top, right, bottom, left  (px)
  face_mask_regions     : list  = [all regions]  # subset of REGION_LABEL_MAP keys
  use_yolo_face_detector: bool  = False           # enable YOLO fast pre-filter

Dependencies (pip install):
  insightface onnxruntime opencv-python numpy
  ultralytics          (only if use_yolo_face_detector=True)
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import insightface

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAME = "ROOP.FACE-SWAPPER"

FaceMaskType = Literal["box", "occlusion", "region"]

# BiSeNet label indices used by the face-parser for region masks
REGION_LABEL_MAP: Dict[str, int] = {
    "skin":          1,
    "left-eyebrow":  2,
    "right-eyebrow": 3,
    "left-eye":      4,
    "right-eye":     5,
    "glasses":       6,
    "nose":         10,
    "mouth":        11,
    "upper-lip":    12,
    "lower-lip":    13,
}

# ---------------------------------------------------------------------------
# Global singletons & thread locks
# ---------------------------------------------------------------------------

FACE_SWAPPER: Optional[Any] = None
YOLO_DETECTOR: Optional[Any] = None
FACE_PARSER: Optional[Any] = None   # BiSeNet ONNX session for region masks

_SWAPPER_LOCK = threading.Lock()
_YOLO_LOCK    = threading.Lock()
_PARSER_LOCK  = threading.Lock()

# ---------------------------------------------------------------------------
# Config helpers  –  read from roop.globals with safe defaults
# ---------------------------------------------------------------------------

def _cfg(attr: str, default: Any) -> Any:
    return getattr(roop.globals, attr, default)


def get_mask_type() -> str:
    return _cfg("face_mask_type", "box")


def get_mask_blur() -> float:
    return float(_cfg("face_mask_blur", 0.3))


def get_mask_padding() -> Tuple[int, int, int, int]:
    return _cfg("face_mask_padding", (0, 0, 0, 0))


def get_mask_regions() -> List[str]:
    return _cfg("face_mask_regions", list(REGION_LABEL_MAP.keys()))


def use_yolo_detector() -> bool:
    return bool(_cfg("use_yolo_face_detector", False))

# ---------------------------------------------------------------------------
# Face-swapper model (inswapper_128)
# ---------------------------------------------------------------------------

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with _SWAPPER_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path("../models/inswapper_128.onnx")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers,
            )
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

# ---------------------------------------------------------------------------
# YOLO face detector  (optional — requires ultralytics + yolov8n-face.pt)
# ---------------------------------------------------------------------------

def get_yolo_detector() -> Optional[Any]:
    """Return a cached YOLOv8-face model, or None if unavailable."""
    global YOLO_DETECTOR
    with _YOLO_LOCK:
        if YOLO_DETECTOR is None:
            try:
                from ultralytics import YOLO  # type: ignore
                model_path = resolve_relative_path("../models/yolov8n-face.pt")
                YOLO_DETECTOR = YOLO(model_path)
            except Exception as exc:
                print(
                    f"[{NAME}] YOLO detector unavailable ({exc}). "
                    "Falling back to InsightFace detector."
                )
                YOLO_DETECTOR = None  # will not retry until next process start
    return YOLO_DETECTOR


def detect_faces_yolo(frame: Frame) -> bool:
    """
    Run YOLO to confirm at least one face is present.
    Returns True when YOLO finds faces (used as a fast pre-filter).
    Falls back to True (no filtering) when YOLO is unavailable.
    """
    detector = get_yolo_detector()
    if detector is None:
        return True   # can't filter, proceed with insightface
    try:
        results = detector(frame, verbose=False)
        for result in results:
            if len(result.boxes) > 0:
                return True
        return False
    except Exception:
        return True   # on error, don't block the swap

# ---------------------------------------------------------------------------
# Face-parser model (BiSeNet via ONNX) for region masks
# ---------------------------------------------------------------------------

def get_face_parser() -> Optional[Any]:
    global FACE_PARSER
    with _PARSER_LOCK:
        if FACE_PARSER is None:
            try:
                import onnxruntime as ort  # type: ignore
                model_path = resolve_relative_path("../models/bisenet_resnet_34.onnx")
                FACE_PARSER = ort.InferenceSession(
                    model_path,
                    providers=roop.globals.execution_providers,
                )
            except Exception as exc:
                print(
                    f"[{NAME}] Face-parser model unavailable ({exc}). "
                    "Region mask will fall back to box mask."
                )
                FACE_PARSER = None
    return FACE_PARSER

# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def _safe_bbox(face: Face, H: int, W: int) -> Tuple[int, int, int, int]:
    """Return a clamped (x1, y1, x2, y2) from a face bounding box."""
    b = face.bbox.astype(int)
    x1 = int(np.clip(b[0], 0, W - 1))
    y1 = int(np.clip(b[1], 0, H - 1))
    x2 = int(np.clip(b[2], 0, W))
    y2 = int(np.clip(b[3], 0, H))
    return x1, y1, x2, y2


def _create_box_mask(
    face: Face,
    H: int,
    W: int,
    blur: float,
    padding: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Simple rectangle mask over the face bbox with optional padding and edge blur.
    Returns float32 (H, W) in [0, 1].
    """
    mask = np.zeros((H, W), dtype=np.float32)
    x1, y1, x2, y2 = _safe_bbox(face, H, W)

    top, right, bottom, left = padding
    y1 = max(0, y1 - top)
    x2 = min(W, x2 + right)
    y2 = min(H, y2 + bottom)
    x1 = max(0, x1 - left)

    if x2 <= x1 or y2 <= y1:
        return mask

    mask[y1:y2, x1:x2] = 1.0

    face_size = max(x2 - x1, y2 - y1)
    kernel_size = max(1, int(face_size * blur))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size > 1:
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return mask


def _create_occlusion_mask(face: Face, frame: Frame) -> np.ndarray:
    """
    Use GrabCut inside the face bbox to isolate foreground skin and exclude
    occluding objects (glasses, hands, scarves …).
    Returns float32 (H, W) in [0, 1].
    """
    H, W = frame.shape[:2]
    full_mask = np.zeros((H, W), dtype=np.float32)
    x1, y1, x2, y2 = _safe_bbox(face, H, W)

    if x2 - x1 < 10 or y2 - y1 < 10:
        # Face region too small for GrabCut — fall back to box
        return _create_box_mask(face, H, W, 0.3, (0, 0, 0, 0))

    face_crop = frame[y1:y2, x1:x2].copy()
    if face_crop.dtype != np.uint8:
        face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)

    gc_mask   = np.zeros(face_crop.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect      = (2, 2, face_crop.shape[1] - 4, face_crop.shape[0] - 4)

    try:
        cv2.grabCut(face_crop, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # GrabCut failure (very small/unusual crop) — fall back to box
        full_mask[y1:y2, x1:x2] = 1.0
        return full_mask

    fg = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1.0, 0.0
    ).astype(np.float32)
    fg = cv2.GaussianBlur(fg, (7, 7), 0)
    full_mask[y1:y2, x1:x2] = fg
    return full_mask


def _create_region_mask(face: Face, frame: Frame, regions: List[str]) -> np.ndarray:
    """
    Run a BiSeNet face-parser to produce a mask covering only the requested
    facial sub-regions (skin, nose, eyes, mouth …).
    Falls back to box mask when the ONNX parser is not available.
    Returns float32 (H, W) in [0, 1].
    """
    H, W = frame.shape[:2]
    parser = get_face_parser()
    if parser is None:
        return _create_box_mask(face, H, W, 0.3, (0, 0, 0, 0))

    x1, y1, x2, y2 = _safe_bbox(face, H, W)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return _create_box_mask(face, H, W, 0.3, (0, 0, 0, 0))

    face_crop = frame[y1:y2, x1:x2]
    INPUT_SIZE = 512
    resized = cv2.resize(face_crop, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb  = (rgb - mean) / std
    blob = rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1, 3, H, W)

    input_name = parser.get_inputs()[0].name
    try:
        outputs = parser.run(None, {input_name: blob})
    except Exception as exc:
        print(f"[{NAME}] Face-parser inference error: {exc}")
        return _create_box_mask(face, H, W, 0.3, (0, 0, 0, 0))

    # outputs[0]: (1, num_classes, 512, 512)
    parse_map = outputs[0][0]                                     # (C, 512, 512)
    label_map = np.argmax(parse_map, axis=0).astype(np.uint8)    # (512, 512)

    region_mask = np.zeros(label_map.shape, dtype=np.uint8)
    for region_name in regions:
        label_idx = REGION_LABEL_MAP.get(region_name)
        if label_idx is not None:
            region_mask[label_map == label_idx] = 255

    # Resize back to original face-crop dimensions
    crop_h, crop_w = y2 - y1, x2 - x1
    region_resized = cv2.resize(
        region_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
    )

    full_mask = np.zeros((H, W), dtype=np.float32)
    full_mask[y1:y2, x1:x2] = region_resized.astype(np.float32) / 255.0
    full_mask = cv2.GaussianBlur(full_mask, (5, 5), 0)
    return full_mask


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    """
    Dispatch to the correct mask builder based on roop.globals.face_mask_type.
    Returns float32 (H, W) in [0, 1].
    """
    H, W    = frame.shape[:2]
    mtype   = get_mask_type()
    blur    = get_mask_blur()
    padding = get_mask_padding()
    regions = get_mask_regions()

    if mtype == "occlusion":
        return _create_occlusion_mask(face, frame)
    if mtype == "region":
        return _create_region_mask(face, frame, regions)
    # Default — box
    return _create_box_mask(face, H, W, blur, padding)

# ---------------------------------------------------------------------------
# Core swap with mask-aware blending
# ---------------------------------------------------------------------------

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    1. Run inswapper to produce a fully-swapped frame.
    2. Build the face mask (box / occlusion / region).
    3. Alpha-blend swapped and original so occluders survive.
    """
    # Step 1 — full inswapper pass
    swapped: Frame = get_face_swapper().get(
        temp_frame, target_face, source_face, paste_back=True
    )

    # Step 2 — face mask
    mask = create_face_mask(target_face, temp_frame)   # float32 (H, W) in [0,1]

    # Step 3 — blending
    mask_3ch = np.stack([mask, mask, mask], axis=-1)   # (H, W, 3)
    blended = (
        swapped.astype(np.float32) * mask_3ch
        + temp_frame.astype(np.float32) * (1.0 - mask_3ch)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Frame-level processing
# ---------------------------------------------------------------------------

def process_frame(
    source_face: Face,
    reference_face: Optional[Face],
    temp_frame: Frame,
) -> Frame:
    if roop.globals.many_faces:
        # Optional YOLO pre-filter: skip expensive insightface pass on blank frames
        if use_yolo_detector() and not detect_faces_yolo(temp_frame):
            return temp_frame
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None],
) -> None:
    source_face    = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face  = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = (
        None
        if roop.globals.many_faces
        else get_one_face(target_frame, roop.globals.reference_face_position)
    )
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face  = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

# ---------------------------------------------------------------------------
# Lifecycle hooks  (identical interface to original)
# ---------------------------------------------------------------------------

def pre_check() -> bool:
    # Models are pre-downloaded by handler.py onto the volume and symlinked.
    # Do NOT call conditional_download here — it re-downloads 529MB on every job.
    # Just verify the model file exists at the resolved path.
    model_path = resolve_relative_path("../models/inswapper_128.onnx")
    if not os.path.exists(model_path):
        update_status(f"inswapper model not found at {model_path}", NAME)
        return False
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False
    if not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status("No face in source path detected.", NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()
