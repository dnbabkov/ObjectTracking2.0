import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from object_tracking_2.segmentators.base_segmentator import BaseSegmentator, SegmentationResult


class SEEMSegmentator(BaseSegmentator):
    def __init__(
        self,
        threshold: float = 0.35,
        min_mask_area: int = 200,
        input_size: int = 640,
        device: str | None = None,
        repos_root: str | None = None,
        repo_dir_name: str = 'Segment-Everything-Everywhere-All-At-Once',
        config_rel_path: str = 'configs/seem/focalt_unicl_lang_demo.yaml',
        weights_rel_path: str = 'seem_focalt_v1.pt',
    ):
        self.threshold = threshold
        self.min_mask_area = min_mask_area
        self.input_size = input_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.min_confidence = 0.85
        self.min_confidence_gap = 0.04

        self.repos_root = Path(repos_root).expanduser().resolve() if repos_root else Path.home() / 'segmentor_git_repos'
        self.repo_root = self.repos_root / repo_dir_name
        self.config_file = self.repo_root / config_rel_path
        self.weights_file = self.repo_root / weights_rel_path

        if not self.repo_root.exists():
            raise FileNotFoundError(f'Не найден репозиторий SEEM: {self.repo_root}')

        if not self.config_file.exists():
            raise FileNotFoundError(f'Не найден конфиг SEEM: {self.config_file}')

        if not self.weights_file.exists():
            raise FileNotFoundError(f'Не найдены веса SEEM: {self.weights_file}')

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        self.model = self._load_model()

        from utils.constants import COCO_PANOPTIC_CLASSES

        with torch.no_grad():
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                COCO_PANOPTIC_CLASSES + ["background"],
                is_eval=True,
            )

    @property
    def name(self) -> str:
        return 'SEEM'

    def _load_model(self):
        import sys

        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from modeling.BaseModel import BaseModel
        from modeling import build_model
        from utils.distributed import init_distributed
        from utils.arguments import load_opt_from_config_files

        opt = load_opt_from_config_files([str(self.config_file)])
        opt = init_distributed(opt)

        model = BaseModel(opt, build_model(opt)).from_pretrained(str(self.weights_file))
        model = model.eval().to(self.device)

        return model

    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]

        resized_rgb = self._resize_keep_ratio(image_rgb, self.input_size)

        start_time = time.perf_counter()
        prediction = self._predict_text_mask(resized_rgb, prompt)
        segmentation_time = time.perf_counter() - start_time

        if prediction is None:
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=segmentation_time,
                metadata={
                    'prompt': prompt,
                    'reason': 'seem_returned_no_mask',
                },
            )

        mask_small = prediction['mask']
        best_confidence = prediction['best_confidence']
        second_confidence = prediction['second_confidence']
        confidence_gap = prediction['confidence_gap']

        if mask_small is None:
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=segmentation_time,
                metadata={
                    'prompt': prompt,
                    'reason': prediction['reason'],
                    'best_confidence': best_confidence,
                    'second_confidence': second_confidence,
                    'confidence_gap': confidence_gap,
                    'min_confidence': self.min_confidence,
                    'min_confidence_gap': self.min_confidence_gap,
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
                    'best_confidence': best_confidence,
                    'second_confidence': second_confidence,
                    'confidence_gap': confidence_gap,
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
                'best_confidence': best_confidence,
                'second_confidence': second_confidence,
                'confidence_gap': confidence_gap,
            },
        )

    def _predict_text_mask(self, image_bgr, prompt: str) -> dict | None:
        import numpy as np
        import torch
        import torch.nn.functional as F

        from PIL import Image
        from utils.visualizer import Visualizer
        from modeling.language.loss import vl_similarity

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        width, height = pil_image.size
        image_np = np.asarray(pil_image)

        images = torch.from_numpy(image_np.copy()).permute(2, 0, 1).to(self.device)

        data = {
            "image": images,
            "height": height,
            "width": width,
            "text": [prompt],
        }
        batch_inputs = [data]

        self.model.model.task_switch["spatial"] = False
        self.model.model.task_switch["visual"] = False
        self.model.model.task_switch["grounding"] = True
        self.model.model.task_switch["audio"] = False

        with torch.no_grad():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    results, image_size, extra = self.model.model.evaluate_demo(batch_inputs)
            else:
                results, image_size, extra = self.model.model.evaluate_demo(batch_inputs)

        pred_masks = results["pred_masks"][0]
        v_emb = results["pred_captions"][0]
        t_emb = extra["grounding_class"]

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

        temperature = self.model.model.sem_seg_head.predictor.lang_encoder.logit_scale

        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

        if out_prob.ndim == 2:
            candidate_scores = out_prob[:, 0].float()
        else:
            candidate_scores = out_prob.float()

        candidate_scores = candidate_scores.detach().cpu()
        candidate_probs = torch.softmax(candidate_scores, dim=0)

        if candidate_probs.numel() == 0:
            return {
                'mask': None,
                'reason': 'no_candidates',
                'best_confidence': None,
                'second_confidence': None,
                'confidence_gap': None,
            }

        sorted_probs, sorted_indices = torch.sort(candidate_probs, descending=True)

        best_confidence = float(sorted_probs[0].item())
        best_index = int(sorted_indices[0].item())

        if sorted_probs.numel() > 1:
            second_confidence = float(sorted_probs[1].item())
            confidence_gap = best_confidence - second_confidence
        else:
            second_confidence = None
            confidence_gap = None

        if best_confidence < self.min_confidence:
            return {
                'mask': None,
                'reason': 'low_confidence',
                'best_confidence': best_confidence,
                'second_confidence': second_confidence,
                'confidence_gap': confidence_gap,
            }

        if confidence_gap is not None and confidence_gap < self.min_confidence_gap:
            return {
                'mask': None,
                'reason': 'small_confidence_gap',
                'best_confidence': best_confidence,
                'second_confidence': second_confidence,
                'confidence_gap': confidence_gap,
            }

        pred_mask = pred_masks[best_index]

        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]
        elif pred_mask.ndim != 2:
            raise RuntimeError(f"Неожиданная форма pred_mask: {tuple(pred_mask.shape)}")

        pred_mask = (
            F.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0] > 0.0
        ).detach().cpu().numpy()

        print(best_confidence)

        return {
            'mask': pred_mask,
            'reason': 'ok',
            'best_confidence': best_confidence,
            'second_confidence': second_confidence,
            'confidence_gap': confidence_gap,
        }

    def _extract_mask_from_outputs(self, outputs: Any) -> np.ndarray | None:
        candidate = None

        if isinstance(outputs, list) and len(outputs) > 0:
            outputs = outputs[0]

        if isinstance(outputs, dict):
            for key in (
                'pred_masks',
                'pred_mask',
                'masks',
                'mask_pred_results',
                'grounding_mask',
                'grounding_masks',
                'sem_seg',
            ):
                if key in outputs:
                    candidate = outputs[key]
                    break
        elif torch.is_tensor(outputs):
            candidate = outputs

        if candidate is None:
            return None

        if torch.is_tensor(candidate):
            tensor = candidate.detach().float().cpu()

            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim == 3:
                tensor = tensor[0]
            if tensor.ndim != 2:
                return None

            prob = torch.sigmoid(tensor).numpy()
            return prob > self.threshold

        if isinstance(candidate, np.ndarray):
            arr = candidate
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim != 2:
                return None
            return arr > self.threshold

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
    def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, center_coords: tuple[int, int] | None) -> np.ndarray:
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