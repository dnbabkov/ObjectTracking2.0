"""
sam2_segmentator.py
===================
Сегментатор на основе SAM2 (Segment Anything Model 2, Meta AI).

Является полным drop-in заменой DinoSAMSegmentator: реализует тот же
интерфейс BaseSegmentator с методом segment(image, prompt, depth).

Отличие от DinoSAM:
  - Вместо SAM v1 + GroundingDINO (два отдельных прогона) используется
    SAM2 в режиме single-frame с предварительной детекцией объекта
    через GroundingDINO (тот же dino_processor/dino_model из dino_sam_segmentator).
  - SAM2 работает в потоковом режиме (camera predictor), что даёт
    значительно более быструю и стабильную сегментацию при повторных
    вызовах для одного и того же объекта (memory bank).
  - При первом вызове (или при смене промпта) — полная инициализация
    через bbox от GroundingDINO.
  - При последующих вызовах — только track(), без повторной детекции DINO.

Зависимости:
    pip install git+https://github.com/facebookresearch/sam2.git
    pip install transformers torch

Модели SAM2 (скачать отдельно):
    https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-checkpoints
    Рекомендуется: sam2.1_hiera_base_plus  (~400 MB)
"""

import gc
import os
import time
from contextlib import nullcontext
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from object_tracking_2.segmentators.base_segmentator import (
    BaseSegmentator,
    SegmentationResult,
)

try:
    from sam2.build_sam import build_sam2_camera_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class SAM2Segmentator(BaseSegmentator):
    """
    Сегментатор на базе SAM2 с memory-based трекингом.

    Первый вызов segment() с новым промптом:
      1. GroundingDINO находит bbox объекта по тексту.
      2. SAM2 инициализируется через этот bbox (add_new_prompt).
      3. Возвращается маска + центр.

    Последующие вызовы с тем же промптом:
      1. SAM2.track() — только прогон memory attention + decoder.
         GroundingDINO не вызывается → latency ~10x ниже.
      2. При потере объекта (низкий score маски) — автоматический
         fallback на reinit через GroundingDINO.
    """

    def __init__(
        self,
        # SAM2
        sam2_cfg: str = 'sam2.1_hiera_base_plus.yaml',
        sam2_checkpoint: str = '/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt',
        # GroundingDINO (для инициализации bbox)
        dino_model_name: str = 'IDEA-Research/grounding-dino-tiny',
        dino_box_threshold: float = 0.35,
        dino_text_threshold: float = 0.25,
        dino_selection_threshold: float = 0.60,
        # SAM2 качество маски
        mask_score_threshold: float = 0.0,   # порог логита маски SAM2
        reinit_score_threshold: float = 0.5, # если mean(sigmoid(logits)) < порога → reinit
        min_mask_area: int = 200,
        # Ограничения маски (как в SEEMSegmentator)
        max_mask_area_ratio: float = 0.45,
        max_bbox_area_ratio: float = 0.70,
        min_bbox_side: int = 10,
    ):
        if not SAM2_AVAILABLE:
            raise ImportError(
                'SAM2 не установлен. Установите: '
                'pip install git+https://github.com/facebookresearch/sam2.git'
            )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_amp = self.device == 'cuda'
        print(f'SAM2Segmentator: device={self.device}')

        # --- Параметры ---
        self.mask_score_threshold   = mask_score_threshold
        self.reinit_score_threshold = reinit_score_threshold
        self.min_mask_area          = min_mask_area
        self.max_mask_area_ratio    = max_mask_area_ratio
        self.max_bbox_area_ratio    = max_bbox_area_ratio
        self.min_bbox_side          = min_bbox_side
        self.dino_box_threshold     = dino_box_threshold
        self.dino_text_threshold    = dino_text_threshold
        self.dino_selection_threshold = dino_selection_threshold

        # --- SAM2 camera predictor ---
        print(f'Загрузка SAM2: {sam2_cfg}')
        self.sam2_predictor = build_sam2_camera_predictor(
            sam2_cfg,
            sam2_checkpoint,
            device=self.device,
        )
        print('SAM2 загружен.')

        # --- GroundingDINO ---
        print(f'Загрузка GroundingDINO: {dino_model_name}')
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_name
        )
        self.dino_model.to(self.device).eval()
        self.dino_model.requires_grad_(False)
        print('GroundingDINO загружен.')

        # --- Состояние трекинга ---
        self._tracking_active    = False   # SAM2 инициализирован и трекает
        self._current_prompt     = None    # последний промпт
        self._frame_idx          = 0       # счётчик кадров для SAM2
        self._obj_id             = 1       # ID объекта в SAM2

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return 'SAM2'

    # ------------------------------------------------------------------
    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()

    def _clear_cuda(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    def _detect_with_dino(
        self, image_rgb: np.ndarray, prompt: str
    ) -> Optional[np.ndarray]:
        """
        Запускает GroundingDINO и возвращает bbox [x1,y1,x2,y2] лучшего
        детектирования или None если ничего не найдено.
        """
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
        return best_box  # [x1, y1, x2, y2]

    # ------------------------------------------------------------------
    def _init_sam2(self, frame_rgb: np.ndarray, bbox_xyxy: np.ndarray):
        """
        Инициализирует SAM2 camera predictor для нового объекта.
        bbox_xyxy: массив [x1, y1, x2, y2]
        """
        self.sam2_predictor.load_first_frame(frame_rgb)
        self._frame_idx = 0

        bbox_input = np.array([bbox_xyxy], dtype=np.float32)
        self.sam2_predictor.add_new_prompt(
            frame_idx=self._frame_idx,
            obj_id=self._obj_id,
            bbox=bbox_input,
        )
        self._tracking_active = True

    # ------------------------------------------------------------------
    def _track_sam2(self, frame_rgb: np.ndarray):
        """
        Прогоняет SAM2.track() на текущем кадре.
        Возвращает (mask_bool, score_mean) или (None, 0.0) при ошибке.
        """
        self._frame_idx += 1
        out_obj_ids, out_logits = self.sam2_predictor.track(frame_rgb)

        if len(out_obj_ids) == 0:
            return None, 0.0

        logits = out_logits[0]           # [1, H, W] или [H, W]
        if logits.ndim == 3:
            logits = logits[0]

        score_mean = float(torch.sigmoid(logits).mean().cpu())
        mask_bool  = (logits > self.mask_score_threshold).cpu().numpy()

        return mask_bool, score_mean

    # ------------------------------------------------------------------
    def _mask_to_result(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        t_total: float,
        metadata: dict,
    ) -> SegmentationResult:
        """Строит SegmentationResult из bool-маски."""
        mask_area = int(np.sum(mask))

        if mask_area < self.min_mask_area or not self._is_reasonable_mask(mask):
            return SegmentationResult(
                vis_image=image_bgr.copy(),
                center_coords=None,
                segmentation_time=t_total,
                metadata={**metadata, 'mask_area': mask_area, 'reason': 'small_mask'},
            )

        center = self._get_center(mask)
        vis    = self._overlay(image_bgr, mask, center)

        return SegmentationResult(
            vis_image=vis,
            center_coords=center,
            segmentation_time=t_total,
            metadata={**metadata, 'mask_area': mask_area},
        )

    # ------------------------------------------------------------------
    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:
        """
        Основной метод — полностью совместим с BaseSegmentator.

        Логика:
          1. Если промпт сменился → сброс трекинга.
          2. Если SAM2 не инициализирован → детекция через DINO → init SAM2.
          3. Если SAM2 активен → track(). Если score низкий → reinit.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t_start   = time.perf_counter()

        # --- Сброс при смене промпта ---
        if prompt != self._current_prompt:
            self._tracking_active = False
            self._current_prompt  = prompt
            self._frame_idx       = 0

        # ── Ветка А: первичная инициализация ──────────────────────────
        if not self._tracking_active:
            t_dino = time.perf_counter()
            bbox = self._detect_with_dino(image_rgb, prompt)
            dino_time = time.perf_counter() - t_dino

            if bbox is None:
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=time.perf_counter() - t_start,
                    metadata={'dino_time': dino_time, 'reason': 'dino_no_detection'},
                )

            self._init_sam2(image_rgb, bbox)

            # Первый кадр — сразу берём маску через track (frame 0 уже загружен)
            mask, score = self._track_sam2(image_rgb)
            t_total = time.perf_counter() - t_start

            if mask is None:
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=t_total,
                    metadata={
                        'dino_time': dino_time,
                        'sam2_score': 0.0,
                        'reason': 'sam2_no_mask_on_init',
                    },
                )

            return self._mask_to_result(
                image, mask, t_total,
                {'mode': 'init', 'dino_time': dino_time, 'sam2_score': score},
            )

        # ── Ветка Б: продолжение трекинга ────────────────────────────
        t_track = time.perf_counter()
        mask, score = self._track_sam2(image_rgb)
        track_time  = time.perf_counter() - t_track
        t_total     = time.perf_counter() - t_start

        # Потеря объекта → reinit на следующем кадре
        if mask is None or score < self.reinit_score_threshold:
            self._tracking_active = False
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=t_total,
                metadata={
                    'mode': 'tracking',
                    'track_time': track_time,
                    'sam2_score': score if score is not None else 0.0,
                    'reason': 'low_score_reinit',
                },
            )

        return self._mask_to_result(
            image, mask, t_total,
            {'mode': 'tracking', 'track_time': track_time, 'sam2_score': score},
        )

    # ------------------------------------------------------------------
    def _is_reasonable_mask(self, mask: np.ndarray) -> bool:
        h, w       = mask.shape[:2]
        total_area = h * w
        mask_area  = int(mask.sum())

        if mask_area <= 0:
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
        pts   = np.column_stack((xs, ys)).astype(np.float32)
        dists = np.sum((pts - np.array([x_med, y_med])) ** 2, axis=1)
        best  = int(np.argmin(dists))
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