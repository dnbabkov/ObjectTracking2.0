"""
dinov2_segmentator.py
=====================
Сегментатор на базе DINOv2 (ViT feature extractor, Meta AI).

Принцип:
  - Первый вызов: GroundingDINO детектирует bbox, из него вырезается
    патч и извлекается DINOv2-эмбеддинг как «шаблон» объекта.
  - Последующие вызовы: dense DINOv2 patch features по всему кадру
    сравниваются с шаблоном cosine similarity; лучший патч — это
    предсказанная позиция объекта.
  - При смене промпта — полный сброс шаблона.

Когда использовать вместо SAM2:
  - SAM2 потерял объект после окклюзии (reinit через DINOv2).
  - Нет GPU (DINOv2-small работает на CPU ~1 FPS, достаточно
    для re-detection раз в несколько секунд).
  - Объект сильно меняет ракурс (DINOv2 инвариантен к ракурсу
    лучше, чем SAM2 memory bank).

Зависимости:
    pip install transformers torch torchvision opencv-python Pillow

Важное замечание о скорости:
  Sliding-window по всему кадру (~3 FPS на GPU).
  Для ускорения используется:
    1. region_of_interest (ROI) — поиск только вокруг последней позиции.
    2. Пирамидальный поиск: сначала крупный шаг, потом уточнение.
"""

import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModel

from object_tracking_2.segmentators.base_segmentator import (
    BaseSegmentator,
    SegmentationResult,
)


class DINOv2Segmentator(BaseSegmentator):
    """
    Zero-shot трекинг через DINOv2 CLS-token cosine similarity.

    Параметры:
        model_name       — HuggingFace model id (dinov2-small/base/large)
        patch_size       — размер патча для extract_feature (px)
        coarse_stride    — шаг грубого поиска (px)
        fine_stride      — шаг точного поиска в ROI (px)
        roi_expand       — коэффициент расширения ROI относительно bbox
        sim_threshold    — минимальный cosine score для обнаружения
        min_mask_area    — минимальная площадь синтетической маски (bbox)
    """

    def __init__(
        self,
        model_name: str = 'facebook/dinov2-small',
        patch_size: int = 224,
        coarse_stride: int = 48,
        fine_stride: int = 16,
        roi_expand: float = 2.5,
        sim_threshold: float = 0.72,
        min_mask_area: int = 200,
        # Начальная детекция через GroundingDINO
        use_dino_grounding: bool = True,
        dino_model_name: str = 'IDEA-Research/grounding-dino-tiny',
        dino_box_threshold: float = 0.35,
        dino_text_threshold: float = 0.25,
        dino_selection_threshold: float = 0.60,
    ):
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.patch_size   = patch_size
        self.coarse_stride = coarse_stride
        self.fine_stride   = fine_stride
        self.roi_expand    = roi_expand
        self.sim_threshold = sim_threshold
        self.min_mask_area = min_mask_area
        print(f'DINOv2Segmentator: device={self.device}')

        # --- DINOv2 ---
        print(f'Загрузка DINOv2: {model_name}')
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model      = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        print('DINOv2 загружен.')

        # --- GroundingDINO для начального bbox ---
        self.use_dino_grounding = use_dino_grounding
        if use_dino_grounding:
            from transformers import AutoModelForZeroShotObjectDetection
            print(f'Загрузка GroundingDINO: {dino_model_name}')
            self.gdino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
            # reuse AutoProcessor for grounding
            from transformers import AutoProcessor as AP
            self.gdino_processor = AP.from_pretrained(dino_model_name)
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                dino_model_name
            )
            self.gdino_model.to(self.device).eval()
            self.gdino_model.requires_grad_(False)
            self.dino_box_threshold       = dino_box_threshold
            self.dino_text_threshold      = dino_text_threshold
            self.dino_sel_threshold       = dino_selection_threshold
            print('GroundingDINO загружен.')

        # --- Состояние ---
        self._template_feat  = None   # torch.Tensor [1, D], нормализован
        self._last_bbox      = None   # (x, y, w, h) последний известный bbox
        self._current_prompt = None

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return 'DINOv2'

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_feat(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """CLS-token из обрезка изображения → нормализованный вектор."""
        pil = PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        pil = pil.resize((self.patch_size, self.patch_size))
        inputs = self.processor(images=pil, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out    = self.model(**inputs)
        feat   = out.last_hidden_state[:, 0, :]      # CLS token [1, D]
        return F.normalize(feat, dim=-1)

    def _detect_initial_bbox(
        self, image_rgb: np.ndarray, prompt: str
    ) -> Optional[np.ndarray]:
        """GroundingDINO → [x1,y1,x2,y2] или None."""
        if not self.use_dino_grounding:
            return None

        pil  = PILImage.fromarray(image_rgb)
        text = str(prompt).strip().lower()
        if not text.endswith('.'):
            text += '.'

        with torch.inference_mode():
            inputs = self.gdino_processor(
                images=pil, text=text, return_tensors='pt'
            ).to(self.device)
            outputs = self.gdino_model(**inputs)

        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.dino_box_threshold,
            text_threshold=self.dino_text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]

        filtered = [
            (box.detach().cpu().numpy(), float(score.item()))
            for box, score in zip(results['boxes'], results['scores'])
            if float(score.item()) >= self.dino_sel_threshold
        ]
        if not filtered:
            return None
        best, _ = max(filtered, key=lambda t: t[1])
        return best  # [x1,y1,x2,y2]

    # ------------------------------------------------------------------
    def _sliding_window_search(
        self,
        frame_bgr: np.ndarray,
        win_w: int,
        win_h: int,
        stride: int,
        search_rect: Optional[tuple] = None,  # (x0,y0,x1,y1) область поиска
    ) -> tuple:
        """
        Скользящее окно в search_rect (или по всему кадру).
        Возвращает (best_bbox_xywh, best_sim).
        """
        H, W = frame_bgr.shape[:2]

        if search_rect is not None:
            sx0, sy0, sx1, sy1 = search_rect
            sx0 = max(0, sx0); sy0 = max(0, sy0)
            sx1 = min(W, sx1); sy1 = min(H, sy1)
        else:
            sx0, sy0, sx1, sy1 = 0, 0, W, H

        best_sim = -1.0
        best_box = None

        for y in range(sy0, sy1 - win_h + 1, stride):
            for x in range(sx0, sx1 - win_w + 1, stride):
                crop = frame_bgr[y:y + win_h, x:x + win_w]
                feat = self._extract_feat(crop)
                sim  = float(torch.sum(self._template_feat * feat).cpu())
                if sim > best_sim:
                    best_sim = sim
                    best_box = (x, y, win_w, win_h)

        return best_box, best_sim

    # ------------------------------------------------------------------
    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:
        """Реализация BaseSegmentator.segment()."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W      = image.shape[:2]
        t_start   = time.perf_counter()

        # Сброс при смене промпта
        if prompt != self._current_prompt:
            self._template_feat  = None
            self._last_bbox      = None
            self._current_prompt = prompt

        # ── Инициализация шаблона ──────────────────────────────────────
        if self._template_feat is None:
            t_dino = time.perf_counter()
            bbox_xyxy = self._detect_initial_bbox(image_rgb, prompt)
            dino_time = time.perf_counter() - t_dino

            if bbox_xyxy is None:
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=time.perf_counter() - t_start,
                    metadata={'dino_time': dino_time, 'reason': 'no_initial_detection'},
                )

            x1, y1, x2, y2 = map(int, bbox_xyxy)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=time.perf_counter() - t_start,
                    metadata={'reason': 'empty_crop'},
                )

            self._template_feat = self._extract_feat(crop)
            w, h = x2 - x1, y2 - y1
            self._last_bbox = (x1, y1, w, h)

        # ── Поиск объекта ─────────────────────────────────────────────
        bx, by, bw, bh = self._last_bbox
        win_w = max(32, bw)
        win_h = max(32, bh)

        # Grубый поиск в расширенном ROI
        roi_cx   = bx + bw // 2
        roi_cy   = by + bh // 2
        expand   = self.roi_expand
        roi_half_w = int(bw * expand)
        roi_half_h = int(bh * expand)
        search_rect = (
            roi_cx - roi_half_w, roi_cy - roi_half_h,
            roi_cx + roi_half_w, roi_cy + roi_half_h,
        )

        coarse_box, coarse_sim = self._sliding_window_search(
            image, win_w, win_h, self.coarse_stride, search_rect
        )

        # Если в ROI ничего — глобальный поиск
        if coarse_box is None or coarse_sim < self.sim_threshold:
            coarse_box, coarse_sim = self._sliding_window_search(
                image, win_w, win_h, self.coarse_stride, None
            )

        if coarse_box is None or coarse_sim < self.sim_threshold:
            t_total = time.perf_counter() - t_start
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=t_total,
                metadata={'best_sim': coarse_sim, 'reason': 'below_threshold'},
            )

        # Точный поиск в окрестности лучшего патча
        fx0 = coarse_box[0] - self.coarse_stride
        fy0 = coarse_box[1] - self.coarse_stride
        fx1 = coarse_box[0] + coarse_box[2] + self.coarse_stride
        fy1 = coarse_box[1] + coarse_box[3] + self.coarse_stride
        fine_box, fine_sim = self._sliding_window_search(
            image, win_w, win_h, self.fine_stride,
            (fx0, fy0, fx1, fy1),
        )

        best_box = fine_box if fine_box is not None and fine_sim >= coarse_sim else coarse_box
        best_sim = fine_sim if fine_box is not None and fine_sim >= coarse_sim else coarse_sim

        if best_sim < self.sim_threshold:
            t_total = time.perf_counter() - t_start
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=t_total,
                metadata={'best_sim': best_sim, 'reason': 'below_threshold_fine'},
            )

        # Обновляем последний известный bbox
        self._last_bbox = best_box
        bx, by, bw, bh  = best_box
        cx = bx + bw // 2
        cy = by + bh // 2
        center_coords = (cx, cy)

        t_total = time.perf_counter() - t_start

        # Визуализация
        vis = image.copy()
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.circle(vis, center_coords, 5, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f'sim={best_sim:.2f}',
            (bx, max(0, by - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        return SegmentationResult(
            vis_image=vis,
            center_coords=center_coords,
            segmentation_time=t_total,
            metadata={
                'best_sim': best_sim,
                'bbox': best_box,
            },
        )