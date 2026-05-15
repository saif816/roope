"""
roop/processors/frame/face_swapper.py
======================================
Drop-in replacement for s0md3v/roop — NO handler changes needed.

xseg.onnx downloads itself automatically on first run, exactly the same
way roop already downloads inswapper_128.onnx.

What this fixes vs original roop:
  - XSeg (DeepFaceLab segmentation model) produces a pixel-accurate face
    mask so the swap never bleeds onto hair, hands, or objects in front.
  - Mask is eroded + feathered — protects jaw/hairline edges.
  - Occluder detection — finds hands/hair crossing the face and restores
    original pixels there instead of the painted-over swap result.
  - Safe fallback to soft elliptical mask if xseg.onnx fails to download.

Only file to change: roop/processors/frame/face_swapper.py
Everything else (handler, run.py, core.py) stays exactly the same.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, List, Optional

import cv2
import insightface
import numpy as np
import onnxruntime

import roop.globals
import roop.processors.frame.core as frame_core
from roop.core import update_status
from roop.face_analyser import get_many_faces, get_one_face, find_similar_face
from roop.face_reference import get_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# ─────────────────────────────────────────────────────────────────────────────
# Tuneable constants
# ─────────────────────────────────────────────────────────────────────────────
XSEG_INPUT_SIZE   = 256    # fixed by the XSeg model
XSEG_THRESHOLD    = 0.10   # lower = include more face pixels in mask
MASK_ERODE_PX     = 8      # pull mask inward — protects hair & jaw edges
MASK_BLUR_PX      = 31     # feather softness (must be odd)
OCCLUDER_DIFF_THR = 25     # pixel diff to call something an occluder
                           #   lower (15) = catches more  |  higher (40) = less aggressive
OCCLUDER_DILATE   = 6      # grow occluder region before subtracting
ELLIPSE_SCALE     = 0.90   # fallback ellipse fraction of face bbox

XSEG_DOWNLOAD_URL = "https://github.com/yakhyo/face-segmentation/releases/download/weights/xseg.onnx"
XSEG_MIN_BYTES    = 60_000_000   # guard against partial downloads (~70 MB expected)

NAME = "ROOP.FACE-SWAPPER"

# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────
_FACE_SWAPPER: Optional[Any]                          = None
_XSEG_SESSION: Optional[onnxruntime.InferenceSession] = None
_XSEG_UNAVAILABLE = False   # True after a failed download — skip retrying every frame
_LOCK             = threading.Lock()

# 5-point canonical template for 256-px face alignment (DFL standard)
_XSEG_TEMPLATE = np.array([
    [0.31556875, 0.46157410],
    [0.68262291, 0.46157410],
    [0.50026249, 0.64050530],
    [0.34947187, 0.82465090],
    [0.65343124, 0.82465090],
], dtype=np.float32) * XSEG_INPUT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _get_face_swapper() -> Any:
    global _FACE_SWAPPER
    with _LOCK:
        if _FACE_SWAPPER is None:
            model_path    = resolve_relative_path('../models/inswapper_128.onnx')
            _FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers,
            )
    return _FACE_SWAPPER


def _get_xseg() -> Optional[onnxruntime.InferenceSession]:
    """
    Load XSeg ONNX session.
    Auto-downloads xseg.onnx on first call if not already present.
    Returns None → falls back to elliptical mask.
    """
    global _XSEG_SESSION, _XSEG_UNAVAILABLE
    with _LOCK:
        if _XSEG_SESSION is not None:
            return _XSEG_SESSION
        if _XSEG_UNAVAILABLE:
            return None

        xseg_path  = resolve_relative_path('../models/xseg.onnx')
        models_dir = resolve_relative_path('../models')

        # ── Auto-download if missing or incomplete ──
        needs_download = (
            not os.path.exists(xseg_path)
            or os.path.getsize(xseg_path) < XSEG_MIN_BYTES
        )
        if needs_download:
            update_status('[XSEG] Downloading xseg.onnx (~70 MB) — one-time download ...', NAME)
            try:
                conditional_download(models_dir, [XSEG_DOWNLOAD_URL])
            except Exception as exc:
                update_status(
                    f'[XSEG] Download failed ({exc}) — using elliptical mask fallback.',
                    NAME,
                )
                _XSEG_UNAVAILABLE = True
                return None

        # Re-check after download attempt
        if not os.path.exists(xseg_path) or os.path.getsize(xseg_path) < XSEG_MIN_BYTES:
            update_status('[XSEG] xseg.onnx still missing after download — using elliptical mask.', NAME)
            _XSEG_UNAVAILABLE = True
            return None

        # ── Load ONNX session ──
        try:
            opts = onnxruntime.SessionOptions()
            opts.log_severity_level = 3          # silence ORT verbose output
            _XSEG_SESSION = onnxruntime.InferenceSession(
                xseg_path,
                sess_options=opts,
                providers=roop.globals.execution_providers,
            )
            update_status('[XSEG] Loaded — occlusion-aware masking active (hands/hair protected).', NAME)
        except Exception as exc:
            update_status(f'[XSEG] Load failed ({exc}) — using elliptical mask fallback.', NAME)
            _XSEG_UNAVAILABLE = True
            return None

    return _XSEG_SESSION


def clear_face_swapper() -> None:
    global _FACE_SWAPPER, _XSEG_SESSION, _XSEG_UNAVAILABLE
    _FACE_SWAPPER     = None
    _XSEG_SESSION     = None
    _XSEG_UNAVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# roop lifecycle hooks — identical interface, nothing changes upstream
# ─────────────────────────────────────────────────────────────────────────────

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(
        download_directory_path,
        ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'],
    )
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()


# ─────────────────────────────────────────────────────────────────────────────
# XSeg inference — returns float32 HxW mask [0..1] in full-frame coordinates
# ─────────────────────────────────────────────────────────────────────────────

def _xseg_mask(frame: Frame, face: Face) -> Optional[np.ndarray]:
    session = _get_xseg()
    if session is None:
        return None
    try:
        h, w = frame.shape[:2]
        lmk  = face.kps   # (5, 2) from insightface

        # Align face chip to 256×256 via similarity transform
        tform = cv2.estimateAffinePartial2D(lmk, _XSEG_TEMPLATE, method=cv2.LMEDS)[0]
        if tform is None:
            return None
        aligned = cv2.warpAffine(
            frame, tform,
            (XSEG_INPUT_SIZE, XSEG_INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
        )

        # Preprocess: BGR→RGB, [0,1], NCHW float32
        inp = aligned[:, :, ::-1].astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]    # 1×3×256×256

        # Run XSeg
        input_name = session.get_inputs()[0].name
        raw        = session.run(None, {input_name: inp})[0]  # 1×1×256×256
        seg        = (raw[0, 0] > XSEG_THRESHOLD).astype(np.float32)

        # Warp mask back to full-frame coordinates
        inv   = cv2.invertAffineTransform(tform)
        full  = cv2.warpAffine(seg, inv, (w, h), flags=cv2.INTER_LINEAR)
        return np.clip(full, 0.0, 1.0)

    except Exception as exc:
        update_status(f'[XSEG] inference error: {exc}', NAME)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Fallback elliptical mask
# ─────────────────────────────────────────────────────────────────────────────

def _ellipse_mask(frame: Frame, face: Face) -> np.ndarray:
    h, w  = frame.shape[:2]
    mask  = np.zeros((h, w), dtype=np.float32)
    b     = face.bbox.astype(int)
    x1, y1 = max(0, b[0]), max(0, b[1])
    x2, y2 = min(w-1, b[2]), min(h-1, b[3])
    cx, cy  = (x1+x2)//2, (y1+y2)//2
    rx      = int((x2-x1)/2 * ELLIPSE_SCALE)
    ry      = int((y2-y1)/2 * ELLIPSE_SCALE)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Mask refinement
# ─────────────────────────────────────────────────────────────────────────────

def _refine_mask(mask: np.ndarray) -> np.ndarray:
    """Erode inward then Gaussian-blur for feathered, edge-safe boundaries."""
    if MASK_ERODE_PX > 0:
        k    = np.ones((MASK_ERODE_PX, MASK_ERODE_PX), np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
    bk   = MASK_BLUR_PX if MASK_BLUR_PX % 2 == 1 else MASK_BLUR_PX + 1
    return cv2.GaussianBlur(mask, (bk, bk), 0)


def _remove_occluders(original: Frame, swapped: Frame, mask: np.ndarray) -> np.ndarray:
    """
    Hands / hair crossing the face force inswapper to paint over them.
    Those pixels appear as large luminance differences vs the original.
    Subtract them from the mask → original pixels show through.
    """
    og   = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sw   = cv2.cvtColor(swapped,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(og - sw)

    occ  = ((diff > OCCLUDER_DIFF_THR) * mask).astype(np.float32)
    if OCCLUDER_DILATE > 0:
        k   = np.ones((OCCLUDER_DILATE, OCCLUDER_DILATE), np.uint8)
        occ = cv2.dilate(occ, k, iterations=1)
        occ = cv2.GaussianBlur(occ, (15, 15), 0)

    return np.clip(mask - occ, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Blend
# ─────────────────────────────────────────────────────────────────────────────

def _blend(original: Frame, swapped: Frame, mask: np.ndarray) -> Frame:
    m   = mask[:, :, np.newaxis]
    out = swapped.astype(np.float32) * m + original.astype(np.float32) * (1.0 - m)
    return out.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Core swap — called per-face per-frame
# ─────────────────────────────────────────────────────────────────────────────

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    1. inswapper_128 raw swap (paste_back=True)
    2. XSeg mask → erode + feather
    3. Remove occluders (hands / hair in front of face)
    4. Alpha blend — original shows everywhere mask=0
    """
    original = temp_frame.copy()
    swapped  = _get_face_swapper().get(
        temp_frame.copy(), target_face, source_face, paste_back=True
    )

    mask = _xseg_mask(original, target_face)
    if mask is None:
        mask = _ellipse_mask(original, target_face)

    mask = _refine_mask(mask)
    mask = _remove_occluders(original, swapped, mask)

    return _blend(original, swapped, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Frame processor interface — identical to original roop, handler untouched
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
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
    for path in temp_frame_paths:
        frame  = cv2.imread(path)
        result = process_frame(source_face, reference_face, frame)
        cv2.imwrite(path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face  = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    ref_face     = get_one_face(target_frame)
    result       = process_frame(source_face, ref_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    frame_core.process_video(source_path, temp_frame_paths, process_frames)
