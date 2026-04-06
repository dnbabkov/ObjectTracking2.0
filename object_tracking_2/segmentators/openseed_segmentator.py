import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg

from object_tracking_2.segmentators.base_segmentator import BaseSegmentator, SegmentationResult


class OpenSeeDSegmentator(BaseSegmentator):
    def __init__(
        self,
        threshold: float = 0.35,
        min_mask_area: int = 200,
        input_size: int = 640,
        device: str | None = None,
        repos_root: str | None = None,
        repo_dir_name: str = 'OpenSeeD',
        config_rel_path: str = 'configs/openseed/openseed_swint_lang.yaml',
        weights_rel_path: str = 'model_state_dict_swint_51.2ap.pt',
    ):
        self.threshold = threshold
        self.min_mask_area = min_mask_area
        self.input_size = input_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.repos_root = Path(repos_root).expanduser().resolve() if repos_root else Path.home() / 'segmentor_git_repos'
        self.repo_root = self.repos_root / repo_dir_name
        self.config_file = self.repo_root / config_rel_path
        self.weights_file = self.repo_root / weights_rel_path

        if not self.repo_root.exists():
            raise FileNotFoundError(f'Не найден репозиторий OpenSeeD: {self.repo_root}')

        if not self.config_file.exists():
            raise FileNotFoundError(f'Не найден конфиг OpenSeeD: {self.config_file}')

        if not self.weights_file.exists():
            raise FileNotFoundError(f'Не найдены веса OpenSeeD: {self.weights_file}')

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        from openseed.architectures.build import build_model

        self._build_model_fn = build_model
        self.cfg = self._load_cfg()
        self.model = self._load_model()

    @property
    def name(self) -> str:
        return 'OpenSeeD'

    def _load_cfg(self):
        cfg = get_cfg()
        with open(self.config_file, 'r', encoding='utf-8') as f:
            raw_cfg = yaml.safe_load(f)

        loaded = self._dict_to_cfgnode(raw_cfg)
        cfg.merge_from_other_cfg(loaded)

        if 'MODEL' not in cfg:
            cfg.MODEL = CN()

        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.WEIGHTS = str(self.weights_file)

        cfg.freeze()
        return cfg

    def _dict_to_cfgnode(self, data: dict[str, Any]) -> CN:
        node = CN(new_allowed=True)

        for key, value in data.items():
            if isinstance(value, dict):
                node[key] = self._dict_to_cfgnode(value)
            elif isinstance(value, list):
                node[key] = [
                    self._dict_to_cfgnode(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                node[key] = value

        return node

    def _load_model(self):
        model = self._build_model_fn(self.cfg)
        model.to(self.device)
        model.eval()

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(str(self.weights_file))
        return model

    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]

        resized_rgb = self._resize_keep_ratio(image_rgb, self.input_size)

        start_time = time.perf_counter()
        mask_small = self._predict_text_mask(resized_rgb, prompt)
        segmentation_time = time.perf_counter() - start_time

        if mask_small is None:
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=segmentation_time,
                metadata={
                    'prompt': prompt,
                    'reason': 'openseed_returned_no_mask',
                },
            )

        mask = cv2.resize(
            mask_small.astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        mask_area = int(mask.sum())
        if mask_area < self.min_mask_area:
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=segmentation_time,
                metadata={
                    'prompt': prompt,
                    'mask_area': mask_area,
                    'threshold': self.threshold,
                },
            )

        center_coords = self.get_center_coordinates(mask)
        vis_image = self._overlay_mask(image, mask, center_coords)

        return SegmentationResult(
            vis_image=vis_image,
            center_coords=center_coords,
            segmentation_time=segmentation_time,
            metadata={
                'prompt': prompt,
                'mask_area': mask_area,
                'threshold': self.threshold,
            },
        )

    def _predict_text_mask(self, image_rgb: np.ndarray, prompt: str) -> np.ndarray | None:
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(self.device) / 255.0

        batched_inputs = [{
            'image': image_tensor,
            'height': image_rgb.shape[0],
            'width': image_rgb.shape[1],
        }]

        class_names = [prompt]
        outputs = None

        with torch.inference_mode():
            for kwargs in (
                {'class_names': class_names, 'task': 'inst_seg'},
                {'class_names': class_names, 'task': 'sem_seg'},
                {'class_names': class_names, 'task': 'panoptic'},
                {'class_names': class_names},
                {},
            ):
                try:
                    outputs = self.model(batched_inputs, **kwargs)
                    if outputs is not None:
                        break
                except Exception:
                    continue

        if outputs is None:
            return None

        return self._extract_best_mask(outputs)

    def _extract_best_mask(self, outputs: Any) -> np.ndarray | None:
        if isinstance(outputs, list) and len(outputs) > 0:
            outputs = outputs[0]

        if not isinstance(outputs, dict):
            return None

        if 'instances' in outputs:
            instances = outputs['instances']
            pred_masks = getattr(instances, 'pred_masks', None)
            scores = getattr(instances, 'scores', None)

            if pred_masks is not None and torch.is_tensor(pred_masks):
                masks = pred_masks.detach().cpu().numpy().astype(bool)

                if len(masks) == 0:
                    return None

                if scores is not None and torch.is_tensor(scores) and len(scores) == len(masks):
                    best_idx = int(scores.argmax().item())
                else:
                    best_idx = int(np.argmax([mask.sum() for mask in masks]))

                return masks[best_idx]

        if 'pred_masks' in outputs:
            pred_masks = outputs['pred_masks']

            if torch.is_tensor(pred_masks):
                tensor = pred_masks.detach().float().cpu()

                if tensor.ndim == 4:
                    tensor = tensor[0]

                if tensor.ndim == 3:
                    probs = torch.sigmoid(tensor).numpy()
                    masks = probs > self.threshold
                    if len(masks) == 0:
                        return None
                    best_idx = int(np.argmax([mask.sum() for mask in masks]))
                    return masks[best_idx]

                if tensor.ndim == 2:
                    prob = torch.sigmoid(tensor).numpy()
                    return prob > self.threshold

        if 'sem_seg' in outputs:
            sem_seg = outputs['sem_seg']
            if torch.is_tensor(sem_seg):
                tensor = sem_seg.detach().float().cpu()
                if tensor.ndim == 3:
                    if tensor.shape[0] > 1:
                        probs = torch.sigmoid(tensor).numpy()
                        masks = probs > self.threshold
                        best_idx = int(np.argmax([mask.sum() for mask in masks]))
                        return masks[best_idx]
                    tensor = tensor[0]
                if tensor.ndim == 2:
                    prob = torch.sigmoid(tensor).numpy()
                    return prob > self.threshold

        return None

    @staticmethod
    def _resize_keep_ratio(image: np.ndarray, max_side: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = max_side / max(h, w)
        if scale >= 1.0:
            return image

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _overlay_mask(
        image_bgr: np.ndarray,
        mask: np.ndarray,
        center_coords: tuple[int, int] | None,
    ) -> np.ndarray:
        out = image_bgr.copy()
        out[mask] = (0, 255, 0)

        if center_coords is not None:
            cv2.circle(out, center_coords, 5, (0, 0, 255), -1)

        return out

    @staticmethod
    def get_center_coordinates(mask: np.ndarray) -> tuple[int, int] | None:
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))
        return x_mean, y_mean