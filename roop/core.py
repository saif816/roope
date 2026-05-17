#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


"""
roop/core.py  —  PATCH SECTION
Copy the block below and paste it inside the  parse_args()  function
of your existing  roop/core.py,  right after the last
`parser.add_argument(...)` call and before `return parser.parse_args()`.

If your project uses run.py to define args instead of roop/core.py,
paste the same block there.
"""

# ── Paste this block into parse_args() ──────────────────────────────────────


def parse_args() -> None:
    """
    Extend the existing argument parser in roop/core.py with new mask flags.
    Add the lines marked NEW below your existing parser.add_argument() calls.
    """
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()

    # ── existing args (keep yours, shown abbreviated) ────────────────────────
    program.add_argument('-s', '--source',     dest='source_path')
    program.add_argument('-t', '--target',     dest='target_path')
    program.add_argument('-o', '--output',     dest='output_path')
    program.add_argument('--frame-processor',  dest='frame_processors', nargs='+')
    program.add_argument('--keep-fps',         dest='keep_fps',    action='store_true')
    program.add_argument('--keep-frames',      dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio',       dest='skip_audio',  action='store_true')
    program.add_argument('--many-faces',       dest='many_faces',  action='store_true')
    program.add_argument('--reference-face-position', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number',  dest='reference_frame_number',  type=int, default=0)
    program.add_argument('--similar-face-distance',   dest='similar_face_distance',   type=float, default=0.85)
    program.add_argument('--temp-frame-format',       dest='temp_frame_format',       default='png')
    program.add_argument('--temp-frame-quality',      dest='temp_frame_quality',      type=int, default=0)
    program.add_argument('--output-video-encoder',    dest='output_video_encoder',    default='libx264')
    program.add_argument('--output-video-quality',    dest='output_video_quality',    type=int, default=35)
    program.add_argument('--max-memory',              dest='max_memory',              type=int)
    program.add_argument('--execution-provider',      dest='execution_providers',     nargs='+')
    program.add_argument('--execution-threads',       dest='execution_threads',       type=int, default=4)
    program.add_argument('--headless',                dest='headless',                action='store_true')
    program.add_argument('--log-level',               dest='log_level',               default='error')

    # ── NEW: face-mask arguments ─────────────────────────────────────────────
    program.add_argument(
        '--face-mask-type',
        dest='face_mask_type',
        choices=['box', 'occlusion', 'region'],
        default='box',
        help='Mask type applied after face swap. '
             'box=simple bbox, occlusion=GrabCut (preserves glasses/hands), '
             'region=BiSeNet parser (swap only selected facial parts).',
    )
    program.add_argument(
        '--face-mask-blur',
        dest='face_mask_blur',
        type=float,
        default=0.3,
        help='Blur amount for box mask edges (0.0=sharp, 1.0=very soft).',
    )
    program.add_argument(
        '--face-mask-padding',
        dest='face_mask_padding',
        default='0 0 0 0',
        help='Box mask padding as "top right bottom left" in pixels.',
    )
    program.add_argument(
        '--face-mask-regions',
        dest='face_mask_regions',
        nargs='+',
        default=None,
        help='Facial regions to include when --face-mask-type region is used. '
             'Choices: skin left-eyebrow right-eyebrow left-eye right-eye '
             'glasses nose mouth upper-lip lower-lip',
    )
    program.add_argument(
        '--use-yolo-face-detector',
        dest='use_yolo_face_detector',
        action='store_true',
        default=False,
        help='Enable YOLOv8-face as a fast face pre-filter (requires ultralytics).',
    )
    # ── end new arguments ─────────────────────────────────────────────────────

    args = program.parse_args()

    # Map parsed args into roop.globals  (existing + new)
    roop.globals.source_path              = args.source_path
    roop.globals.target_path              = args.target_path
    roop.globals.output_path              = args.output_path
    roop.globals.headless                 = args.headless
    roop.globals.frame_processors         = args.frame_processors
    roop.globals.keep_fps                 = args.keep_fps
    roop.globals.keep_frames              = args.keep_frames
    roop.globals.skip_audio               = args.skip_audio
    roop.globals.many_faces               = args.many_faces
    roop.globals.reference_face_position  = args.reference_face_position
    roop.globals.reference_frame_number   = args.reference_frame_number
    roop.globals.similar_face_distance    = args.similar_face_distance
    roop.globals.temp_frame_format        = args.temp_frame_format
    roop.globals.temp_frame_quality       = args.temp_frame_quality
    roop.globals.output_video_encoder     = args.output_video_encoder
    roop.globals.output_video_quality     = args.output_video_quality
    roop.globals.max_memory               = args.max_memory
    roop.globals.execution_providers      = decode_execution_providers(args.execution_providers)
    roop.globals.execution_threads        = args.execution_threads
    roop.globals.log_level                = args.log_level

    # ── NEW globals ──────────────────────────────────────────────────────────
    roop.globals.face_mask_type           = args.face_mask_type

    roop.globals.face_mask_blur           = max(0.0, min(1.0, args.face_mask_blur))

    # Parse padding string "top right bottom left" → tuple of ints
    try:
        parts = [int(x) for x in args.face_mask_padding.split()]
        if len(parts) == 1:
            parts = parts * 4
        elif len(parts) == 2:
            parts = [parts[0], parts[1], parts[0], parts[1]]
        elif len(parts) != 4:
            parts = [0, 0, 0, 0]
    except (ValueError, AttributeError):
        parts = [0, 0, 0, 0]
    roop.globals.face_mask_padding        = tuple(parts)

    roop.globals.face_mask_regions        = args.face_mask_regions or [
        "skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye",
        "nose", "mouth", "upper-lip", "lower-lip",
    ]

    roop.globals.use_yolo_face_detector   = args.use_yolo_face_detector
    # ── end new globals ───────────────────────────────────────────────────────


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)


def start() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path):
            destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        # process frame
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        # validate image
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if predict_video(roop.globals.target_path):
        destroy()
    update_status('Creating temporary resources...')
    create_temp(roop.globals.target_path)
    # extract frames
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(roop.globals.target_path, fps)
    else:
        update_status('Extracting frames with 30 FPS...')
        extract_frames(roop.globals.target_path)
    # process frame
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        return
    # create video
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(roop.globals.target_path)
    # handle audio
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
        update_status('Skipping audio...')
    else:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    # clean temp
    update_status('Cleaning temporary resources...')
    clean_temp(roop.globals.target_path)
    # validate video
    if is_video(roop.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if roop.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()
