import gc
import os
import cv2
import time
import shutil
import tempfile
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Deque, Tuple, List

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from object_tracking_2.segmentators.base_segmentator import (
    BaseSegmentator,
    SegmentationResult,
)

try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    SAM2_IMPORT_ERROR = None
except Exception as e:
    SAM2_AVAILABLE = False
    SAM2_IMPORT_ERROR = e


class SAM2Segmentator(BaseSegmentator):
    """
    SAM2 segmentator with two stages:

    Stage A (initialization):
      - GroundingDINO finds bbox on current frame.
      - SAM2ImagePredictor refines bbox into a mask.
      - The current frame becomes the first frame of a mini-video buffer.

    Stage B (tracking on mini-video):
      - Frames are accumulated in a sliding buffer.
      - Once the buffer is large enough, we dump frames to a temp directory.
      - Official SAM2 video predictor is run on this short video clip.
      - We prompt frame 0 of the clip with the last known bbox.
      - We propagate through the buffer and return the mask on the LAST frame.

    This is compatible with official SAM2 API:
      - images: build_sam2 + SAM2ImagePredictor
      - videos: build_sam2_video_predictor + init_state + add_new_points_or_box + propagate_in_video
    """

    def __init__(
        self,
        # --- SAM2 paths ---
        sam2_cfg: str = os.environ.get(
            'SAM2_CFG',
            'configs/sam2.1/sam2.1_hiera_b+.yaml',
        ),
        sam2_checkpoint: str = os.environ.get(
            'SAM2_CKPT',
            os.path.expanduser('~/models/sam2/sam2.1_hiera_base_plus.pt'),
        ),

        # --- GroundingDINO ---
        dino_model_name: str = 'IDEA-Research/grounding-dino-tiny',
        dino_box_threshold: float = 0.35,
        dino_text_threshold: float = 0.25,
        dino_selection_threshold: float = 0.60,

        # --- Buffer / mini-video ---
        video_buffer_size: int = 8,
        min_video_frames: int = 4,
        reinit_every_n_frames: int = 10,
        track_score_threshold: float = 0.0,

        # --- Mask filtering ---
        min_mask_area: int = 200,
        max_mask_area_ratio: float = 0.45,
        max_bbox_area_ratio: float = 0.70,
        min_bbox_side: int = 10,
    ):
        if not SAM2_AVAILABLE:
            raise ImportError(f'SAM2 import failed: {SAM2_IMPORT_ERROR}')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_amp = self.device == 'cuda'

        self.sam2_cfg = sam2_cfg
        self.sam2_checkpoint = sam2_checkpoint

        self.dino_box_threshold = dino_box_threshold
        self.dino_text_threshold = dino_text_threshold
        self.dino_selection_threshold = dino_selection_threshold

        self.video_buffer_size = max(2, int(video_buffer_size))
        self.min_video_frames = max(2, int(min_video_frames))
        self.reinit_every_n_frames = max(1, int(reinit_every_n_frames))
        self.track_score_threshold = track_score_threshold

        self.min_mask_area = min_mask_area
        self.max_mask_area_ratio = max_mask_area_ratio
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.min_bbox_side = min_bbox_side

        print(f'SAM2Segmentator: device={self.device}')
        print(f'SAM2 cfg={self.sam2_cfg}')
        print(f'SAM2 ckpt={self.sam2_checkpoint}')

        # --- Official SAM2 image predictor ---
        image_model = build_sam2(
            config_file=self.sam2_cfg,
            ckpt_path=self.sam2_checkpoint,
            device=self.device,
        )
        self.image_predictor = SAM2ImagePredictor(image_model)

        # --- Official SAM2 video predictor ---
        # If your installed SAM2 version supports vos_optimized, you can enable it here.
        self.video_predictor = build_sam2_video_predictor(
            config_file=self.sam2_cfg,
            ckpt_path=self.sam2_checkpoint,
            device=self.device,
        )

        # --- GroundingDINO ---
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_name
        )
        self.dino_model.to(self.device).eval()
        self.dino_model.requires_grad_(False)

        # --- Tracking state ---
        self._current_prompt: Optional[str] = None
        self._tracking_active: bool = False
        self._frames_since_reinit: int = 0

        # sliding window of RGB frames
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.video_buffer_size)

        # bbox in XYXY format, tied to the FIRST frame currently in buffer
        self._anchor_box_xyxy: Optional[np.ndarray] = None

        # last known bbox/mask on current frame
        self._last_box_xyxy: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return 'SAM2'

    # ------------------------------------------------------------------
    # low-level utils
    # ------------------------------------------------------------------

    def _autocast(self):
        if self.use_amp:
            # bf16 is typically what official README uses on CUDA
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return nullcontext()

    def _clear_cuda(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def _reset_tracking(self):
        self._tracking_active = False
        self._frames_since_reinit = 0
        self._frame_buffer.clear()
        self._anchor_box_xyxy = None
        self._last_box_xyxy = None
        self._last_mask = None

    # ------------------------------------------------------------------
    # GroundingDINO init
    # ------------------------------------------------------------------

    def _detect_with_dino(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> Optional[np.ndarray]:
        image_pil = PILImage.fromarray(image_rgb)
        text = str(prompt).strip().lower()
        if not text.endswith('.'):
            text += '.'

        with torch.inference_mode():
            with self._autocast():
                inputs = self.dino_processor(
                    images=image_pil,
                    text=text,
                    return_tensors='pt',
                ).to(self.device)
                outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.dino_box_threshold,
            text_threshold=self.dino_text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        del outputs, inputs
        self._clear_cuda()

        filtered = [
            (box.detach().cpu().numpy(), float(score.item()))
            for box, score in zip(results['boxes'], results['scores'])
            if float(score.item()) >= self.dino_selection_threshold
        ]

        if not filtered:
            return None

        best_box, _ = max(filtered, key=lambda item: item[1])
        return best_box.astype(np.float32)

    # ------------------------------------------------------------------
    # SAM2 image step
    # ------------------------------------------------------------------

    def _segment_with_image_predictor(
        self,
        image_rgb: np.ndarray,
        box_xyxy: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Returns (mask_bool, score).
        """
        with torch.inference_mode():
            with self._autocast():
                self.image_predictor.set_image(image_rgb)
                masks, scores, _ = self.image_predictor.predict(
                    box=box_xyxy[None, :],
                    multimask_output=False,
                )

        if masks is None or len(masks) == 0:
            return None, 0.0

        # common case: masks shape is (1, H, W)
        mask = masks[0].astype(bool)
        score = float(scores[0]) if len(scores) > 0 else 0.0
        return mask, score

    # ------------------------------------------------------------------
    # mini-video via official SAM2 video predictor
    # ------------------------------------------------------------------

    def _write_buffer_to_temp_video_dir(
        self,
        frames_rgb: List[np.ndarray],
    ) -> str:
        """
        Official SAM2 video predictor accepts a video / frame source.
        The simplest robust path in Python tooling is to dump a short frame
        sequence into a temp directory and pass it to init_state(...).
        """
        temp_dir = tempfile.mkdtemp(prefix='sam2_minivideo_')
        for i, frame_rgb in enumerate(frames_rgb):
            # save as JPEG frames 00000.jpg, 00001.jpg, ...
            out_path = os.path.join(temp_dir, f'{i:05d}.jpg')
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, frame_bgr)
        return temp_dir

    def _track_on_buffer_with_video_predictor(
        self,
        frames_rgb: List[np.ndarray],
        init_box_xyxy: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Runs official video predictor on the whole mini-video buffer and
        returns mask for the LAST frame in the buffer.

        We prompt frame_idx=0 of the clip using init_box_xyxy.
        """
        temp_dir = self._write_buffer_to_temp_video_dir(frames_rgb)

        try:
            with torch.inference_mode():
                with self._autocast():
                    state = self.video_predictor.init_state(video_path=temp_dir)

                    # obj_id can be any positive int
                    self.video_predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=1,
                        box=init_box_xyxy.astype(np.float32),
                    )

                    last_mask = None
                    last_score = 0.0

                    for out_frame_idx, out_obj_ids, out_mask_logits in \
                            self.video_predictor.propagate_in_video(state):
                        # We only care about the last frame in the buffer.
                        if out_frame_idx != len(frames_rgb) - 1:
                            continue

                        if out_mask_logits is None or len(out_mask_logits) == 0:
                            break

                        logits = out_mask_logits[0]
                        if hasattr(logits, 'detach'):
                            logits_np = logits.detach().float().cpu().numpy()
                        else:
                            logits_np = np.asarray(logits, dtype=np.float32)

                        if logits_np.ndim == 3:
                            logits_np = logits_np[0]

                        last_mask = logits_np > self.track_score_threshold
                        last_score = float(1.0 / (1.0 + np.exp(-logits_np)).mean())
                        break

                    return last_mask, last_score

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self._clear_cuda()

    # ------------------------------------------------------------------
    # geometry / mask helpers
    # ------------------------------------------------------------------

    def _mask_to_box_xyxy(self, mask: np.ndarray) -> Optional[np.ndarray]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _is_reasonable_mask(self, mask: np.ndarray) -> bool:
        h, w = mask.shape[:2]
        total_area = h * w
        mask_area = int(mask.sum())

        if mask_area <= 0:
            return False
        if mask_area < self.min_mask_area:
            return False
        if mask_area / float(total_area) > self.max_mask_area_ratio:
            return False

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return False

        bw = int(xs.max()) - int(xs.min()) + 1
        bh = int(ys.max()) - int(ys.min()) + 1

        if bw < self.min_bbox_side or bh < self.min_bbox_side:
            return False
        if (bw * bh) / float(total_area) > self.max_bbox_area_ratio:
            return False

        return True

    @staticmethod
    def _get_center(mask: np.ndarray) -> Optional[tuple]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None

        x_med = float(np.median(xs))
        y_med = float(np.median(ys))
        pts = np.column_stack((xs, ys)).astype(np.float32)
        dists = np.sum((pts - np.array([x_med, y_med])) ** 2, axis=1)
        best = int(np.argmin(dists))
        return int(pts[best, 0]), int(pts[best, 1])

    @staticmethod
    def _overlay(
        image_bgr: np.ndarray,
        mask: np.ndarray,
        center: Optional[tuple],
    ) -> np.ndarray:
        out = image_bgr.copy()
        out[mask] = (0, 255, 0)

        ys, xs = np.where(mask)
        if len(xs) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if center is not None:
            cv2.circle(out, center, 5, (0, 0, 255), -1)

        return out

    def _make_result(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray],
        t_total: float,
        metadata: dict,
    ) -> SegmentationResult:
        if mask is None or not self._is_reasonable_mask(mask):
            return SegmentationResult(
                vis_image=image_bgr.copy(),
                center_coords=None,
                segmentation_time=t_total,
                metadata=metadata,
            )

        center = self._get_center(mask)
        vis = self._overlay(image_bgr, mask, center)
        return SegmentationResult(
            vis_image=vis,
            center_coords=center,
            segmentation_time=t_total,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t_start = time.perf_counter()

        # prompt changed -> reset state
        if prompt != self._current_prompt:
            self._current_prompt = prompt
            self._reset_tracking()

        # ==========================================================
        # A. INIT: detect via DINO, refine via SAM2 image predictor
        # ==========================================================
        if not self._tracking_active:
            t_dino = time.perf_counter()
            init_box = self._detect_with_dino(image_rgb, prompt)
            dino_time = time.perf_counter() - t_dino

            if init_box is None:
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=time.perf_counter() - t_start,
                    metadata={
                        'mode': 'init',
                        'reason': 'dino_no_detection',
                        'dino_time': dino_time,
                    },
                )

            t_img = time.perf_counter()
            init_mask, init_score = self._segment_with_image_predictor(
                image_rgb,
                init_box,
            )
            img_time = time.perf_counter() - t_img

            if init_mask is None or not self._is_reasonable_mask(init_mask):
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=time.perf_counter() - t_start,
                    metadata={
                        'mode': 'init',
                        'reason': 'sam2_image_no_mask',
                        'dino_time': dino_time,
                        'image_time': img_time,
                        'sam2_score': init_score,
                    },
                )

            refined_box = self._mask_to_box_xyxy(init_mask)
            if refined_box is None:
                refined_box = init_box

            # start mini-video buffer from the init frame
            self._frame_buffer.clear()
            self._frame_buffer.append(image_rgb.copy())

            self._anchor_box_xyxy = refined_box.copy()
            self._last_box_xyxy = refined_box.copy()
            self._last_mask = init_mask.copy()
            self._tracking_active = True
            self._frames_since_reinit = 0

            t_total = time.perf_counter() - t_start
            return self._make_result(
                image,
                init_mask,
                t_total,
                {
                    'mode': 'init_image',
                    'dino_time': dino_time,
                    'image_time': img_time,
                    'sam2_score': init_score,
                    'buffer_size': len(self._frame_buffer),
                },
            )

        # ==========================================================
        # B. TRACK: accumulate mini-video and run official video predictor
        # ==========================================================
        self._frame_buffer.append(image_rgb.copy())
        self._frames_since_reinit += 1

        # Until buffer is large enough, fall back to per-frame image predictor
        if len(self._frame_buffer) < self.min_video_frames:
            fallback_box = self._last_box_xyxy
            if fallback_box is None:
                fallback_box = self._anchor_box_xyxy

            if fallback_box is None:
                self._reset_tracking()
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=time.perf_counter() - t_start,
                    metadata={'mode': 'warmup', 'reason': 'missing_fallback_box'},
                )

            mask, score = self._segment_with_image_predictor(image_rgb, fallback_box)
            if mask is None or not self._is_reasonable_mask(mask):
                self._reset_tracking()
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=time.perf_counter() - t_start,
                    metadata={
                        'mode': 'warmup_image',
                        'reason': 'fallback_image_failed',
                        'sam2_score': score,
                    },
                )

            self._last_mask = mask.copy()
            self._last_box_xyxy = self._mask_to_box_xyxy(mask)

            t_total = time.perf_counter() - t_start
            return self._make_result(
                image,
                mask,
                t_total,
                {
                    'mode': 'warmup_image',
                    'sam2_score': score,
                    'buffer_size': len(self._frame_buffer),
                },
            )

        # if the sliding window moved, anchor box still refers to first frame
        # in the window only if we kept it synchronized. We choose a simple,
        # robust policy: periodically re-initialize from current frame.
        if self._frames_since_reinit >= self.reinit_every_n_frames:
            self._tracking_active = False
            return self.segment(image, prompt, depth)

        if self._anchor_box_xyxy is None:
            # no valid anchor for video prompt -> full reinit next call
            self._tracking_active = False
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=time.perf_counter() - t_start,
                metadata={
                    'mode': 'video',
                    'reason': 'missing_anchor_box_reinit',
                },
            )

        t_video = time.perf_counter()
        video_mask, video_score = self._track_on_buffer_with_video_predictor(
            list(self._frame_buffer),
            self._anchor_box_xyxy,
        )
        video_time = time.perf_counter() - t_video

        if video_mask is None or not self._is_reasonable_mask(video_mask):
            # failed on mini-video -> force DINO+image re-init on next frame
            self._tracking_active = False
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=time.perf_counter() - t_start,
                metadata={
                    'mode': 'video',
                    'reason': 'video_predictor_failed_reinit',
                    'video_time': video_time,
                    'sam2_score': video_score,
                },
            )

        self._last_mask = video_mask.copy()
        self._last_box_xyxy = self._mask_to_box_xyxy(video_mask)

        # IMPORTANT:
        # because the sliding window is moving, the prompt for frame 0 of the
        # next mini-video window is no longer guaranteed to be exact.
        # To keep logic simple and stable, we re-anchor the buffer from CURRENT frame
        # after each successful video step.
        self._frame_buffer.clear()
        self._frame_buffer.append(image_rgb.copy())
        if self._last_box_xyxy is not None:
            self._anchor_box_xyxy = self._last_box_xyxy.copy()
        self._frames_since_reinit = 0

        t_total = time.perf_counter() - t_start
        return self._make_result(
            image,
            video_mask,
            t_total,
            {
                'mode': 'mini_video',
                'video_time': video_time,
                'sam2_score': video_score,
                'buffer_size': len(self._frame_buffer),
            },
        )