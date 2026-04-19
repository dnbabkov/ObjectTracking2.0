import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from detectron2.data import MetadataCatalog

from object_tracking_2.segmentators.base_segmentator import BaseSegmentator, SegmentationResult


class OpenSeeDSegmentator(BaseSegmentator):
    def __init__(
        self,
        score_threshold: float = 0.20,
        mask_threshold: float = 0.35,
        min_mask_area: int = 200,
        max_mask_area_ratio: float = 0.60,
        min_bbox_side: int = 8,
        input_size: int = 640,
        device: str | None = None,
        repos_root: str | None = None,
        repo_dir_name: str = 'OpenSeeD',
        config_rel_path: str = 'configs/openseed/openseed_swint_lang.yaml',
        weights_rel_path: str = 'model_state_dict_swint_51.2ap.pt',
    ):
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.min_mask_area = min_mask_area
        self.max_mask_area_ratio = max_mask_area_ratio
        self.min_bbox_side = min_bbox_side
        self.input_size = input_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.repos_root = (
            Path(repos_root).expanduser().resolve()
            if repos_root
            else Path.home() / 'segmentor_git_repos'
        )
        self.repo_root = self.repos_root / repo_dir_name
        self.config_file = self.repo_root / config_rel_path
        self.weights_file = self.repo_root / weights_rel_path

        self._last_vocab: tuple[str, ...] | None = None
        self._metadata_name: str | None = None

        self._color_words = {
            'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'gray', 'grey', 'brown', 'dark', 'light',
        }

        if not self.repo_root.exists():
            raise FileNotFoundError(f'Не найден репозиторий OpenSeeD: {self.repo_root}')

        if not self.config_file.exists():
            raise FileNotFoundError(f'Не найден конфиг OpenSeeD: {self.config_file}')

        if not self.weights_file.exists():
            raise FileNotFoundError(f'Не найдены веса OpenSeeD: {self.weights_file}')

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        self.opt = self._load_opt()
        self.model = self._load_model()

    @property
    def name(self) -> str:
        return 'OpenSeeD'

    def _load_opt(self) -> dict[str, Any]:
        with open(self.config_file, 'r', encoding='utf-8') as f:
            opt = yaml.safe_load(f)

        if not isinstance(opt, dict):
            raise RuntimeError(f'OpenSeeD config must be dict, got {type(opt)}')

        opt['WEIGHT'] = str(self.weights_file)
        return opt

    def _load_model(self):
        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from openseed.BaseModel import BaseModel
        from openseed import build_model

        model = BaseModel(
            self.opt,
            build_model(self.opt),
        ).from_pretrained(str(self.weights_file))

        model.eval()
        model.to(self.device)

        for param in model.parameters():
            param.requires_grad_(False)

        return model

    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]

        resized_rgb = self._resize_keep_ratio(image_rgb, self.input_size)
        vocab = self._build_vocabulary(prompt)
        self._configure_vocabulary(vocab)

        start_time = time.perf_counter()
        match = self._predict_best_match(resized_rgb, vocab)
        segmentation_time = time.perf_counter() - start_time

        if match is None:
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=segmentation_time,
                metadata={
                    'prompt': prompt,
                    'vocab': vocab,
                    'reason': 'openseed_returned_no_mask',
                },
            )

        mask_small = match['mask']

        if mask_small.shape[:2] != (original_h, original_w):
            mask = cv2.resize(
                mask_small.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask = mask_small.astype(bool)

        if not self._is_reasonable_mask(mask):
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=segmentation_time,
                metadata={
                    'prompt': prompt,
                    'vocab': vocab,
                    'reason': 'unreasonable_mask',
                    'label': match.get('label'),
                    'score': match.get('score'),
                },
            )

        mask_area = int(mask.sum())
        if mask_area < self.min_mask_area:
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=segmentation_time,
                metadata={
                    'prompt': prompt,
                    'vocab': vocab,
                    'mask_area': mask_area,
                    'label': match.get('label'),
                    'score': match.get('score'),
                    'score_threshold': self.score_threshold,
                    'mask_threshold': self.mask_threshold,
                },
            )

        center_coords = self.get_center_coordinates(mask)
        vis_image = self._overlay_mask(
            image_bgr=image,
            mask=mask,
            center_coords=center_coords,
            label=match.get('label'),
            score=match.get('score'),
        )

        return SegmentationResult(
            vis_image=vis_image,
            center_coords=center_coords,
            segmentation_time=segmentation_time,
            metadata={
                'prompt': prompt,
                'vocab': vocab,
                'mask_area': mask_area,
                'label': match.get('label'),
                'score': match.get('score'),
                'score_threshold': self.score_threshold,
                'mask_threshold': self.mask_threshold,
            },
        )

    def _build_vocabulary(self, prompt: str) -> list[str]:
        text = str(prompt).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-]', '', text).strip()

        if not text:
            return ['object']

        vocab = [text]

        words = text.split()
        simplified_words = [w for w in words if w not in self._color_words]
        simplified_text = ' '.join(simplified_words).strip()

        if simplified_text and simplified_text != text:
            vocab.append(simplified_text)

        unique_vocab = []
        seen = set()
        for item in vocab:
            if item not in seen:
                seen.add(item)
                unique_vocab.append(item)

        return unique_vocab

    def _stable_color(self, text: str) -> list[int]:
        digest = hashlib.md5(text.encode('utf-8')).digest()
        return [int(digest[0]), int(digest[1]), int(digest[2])]

    def _configure_vocabulary(self, thing_classes: list[str]) -> None:
        vocab_tuple = tuple(thing_classes)
        if self._last_vocab == vocab_tuple:
            return

        vocab_key = hashlib.md5('||'.join(thing_classes).encode('utf-8')).hexdigest()[:12]
        metadata_name = f'openseed_ros_demo__{vocab_key}'

        metadata = MetadataCatalog.get(metadata_name)
        metadata.set(
            thing_classes=thing_classes,
            thing_colors=[self._stable_color(cls) for cls in thing_classes],
            thing_dataset_id_to_contiguous_id={i: i for i in range(len(thing_classes))},
            stuff_classes=[],
            stuff_colors=[],
            stuff_dataset_id_to_contiguous_id={},
        )

        predictor = self.model.model.sem_seg_head.predictor
        predictor.lang_encoder.get_text_embeddings(thing_classes, is_eval=True)

        self.model.model.metadata = metadata
        self.model.model.sem_seg_head.num_classes = len(thing_classes)

        self._metadata_name = metadata_name
        self._last_vocab = vocab_tuple

    def _predict_best_match(
        self,
        image_rgb: np.ndarray,
        vocab: list[str],
    ) -> dict[str, Any] | None:
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image_rgb))
            .permute(2, 0, 1)
            .float()
            .to(self.device)
        )

        batch_inputs = [{
            'image': image_tensor,
            'height': image_rgb.shape[0],
            'width': image_rgb.shape[1],
        }]

        with torch.inference_mode():
            outputs = self.model.forward(batch_inputs)

        if isinstance(outputs, list) and len(outputs) > 0:
            outputs = outputs[0]

        if not isinstance(outputs, dict):
            return None

        match = self._extract_from_instances(outputs, vocab)
        if match is not None and self._is_reasonable_mask(match['mask']):
            return match

        match = self._extract_from_panoptic(outputs, vocab)
        if match is not None and self._is_reasonable_mask(match['mask']):
            return match

        match = self._extract_from_query_outputs(outputs, vocab)
        if match is not None and self._is_reasonable_mask(match['mask']):
            return match

        return None

    def _extract_from_instances(
        self,
        outputs: dict[str, Any],
        vocab: list[str],
    ) -> dict[str, Any] | None:
        instances = outputs.get('instances')
        if instances is None:
            return None

        pred_masks = getattr(instances, 'pred_masks', None)
        scores = getattr(instances, 'scores', None)
        pred_classes = getattr(instances, 'pred_classes', None)

        if pred_masks is None or not torch.is_tensor(pred_masks):
            return None

        masks_np = pred_masks.detach().cpu().numpy().astype(bool)
        if len(masks_np) == 0:
            return None

        scores_np = None
        if scores is not None and torch.is_tensor(scores):
            scores_np = scores.detach().cpu().numpy()

        classes_np = None
        if pred_classes is not None and torch.is_tensor(pred_classes):
            classes_np = pred_classes.detach().cpu().numpy()

        candidates: list[dict[str, Any]] = []

        for idx, mask in enumerate(masks_np):
            area = int(mask.sum())
            if area <= 0:
                continue

            score = float(scores_np[idx]) if scores_np is not None else 0.0
            if scores_np is not None and score < self.score_threshold:
                continue

            cls_idx = int(classes_np[idx]) if classes_np is not None else 0
            if cls_idx < 0 or cls_idx >= len(vocab):
                continue

            candidates.append({
                'mask': mask,
                'score': score,
                'label': vocab[cls_idx],
                'area': area,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item['score'], item['area']), reverse=True)
        return candidates[0]

    def _extract_from_panoptic(
        self,
        outputs: dict[str, Any],
        vocab: list[str],
    ) -> dict[str, Any] | None:
        panoptic = outputs.get('panoptic_seg')
        if panoptic is None or not isinstance(panoptic, (tuple, list)) or len(panoptic) != 2:
            return None

        pano_seg, segments_info = panoptic
        if not torch.is_tensor(pano_seg) or segments_info is None:
            return None

        pano_seg_np = pano_seg.detach().cpu().numpy()

        best: dict[str, Any] | None = None
        for info in segments_info:
            category_id = int(info.get('category_id', -1))
            if category_id < 0 or category_id >= len(vocab):
                continue

            segment_id = int(info.get('id', -1))
            if segment_id < 0:
                continue

            mask = pano_seg_np == segment_id
            area = int(mask.sum())
            if area <= 0:
                continue

            score = float(info.get('score', 1.0))
            if score < self.score_threshold:
                continue

            candidate = {
                'mask': mask,
                'score': score,
                'label': vocab[category_id],
                'area': area,
            }

            if best is None or (candidate['score'], candidate['area']) > (best['score'], best['area']):
                best = candidate

        return best

    def _extract_from_query_outputs(
        self,
        outputs: dict[str, Any],
        vocab: list[str],
    ) -> dict[str, Any] | None:
        pred_masks = outputs.get('pred_masks')
        pred_logits = outputs.get('pred_logits')

        if pred_masks is None or pred_logits is None:
            return None
        if not torch.is_tensor(pred_masks) or not torch.is_tensor(pred_logits):
            return None

        masks_tensor = pred_masks.detach().float().cpu()
        logits_tensor = pred_logits.detach().float().cpu()

        if masks_tensor.ndim == 4:
            masks_tensor = masks_tensor[0]
        if logits_tensor.ndim == 3:
            logits_tensor = logits_tensor[0]

        if masks_tensor.ndim != 3 or logits_tensor.ndim != 2:
            return None
        if masks_tensor.shape[0] != logits_tensor.shape[0]:
            return None

        class_count = min(len(vocab), logits_tensor.shape[1])
        if class_count <= 0:
            return None

        probs = torch.softmax(logits_tensor, dim=-1)[:, :class_count]
        best_scores, best_classes = probs.max(dim=1)

        candidates: list[dict[str, Any]] = []

        for idx in range(masks_tensor.shape[0]):
            score = float(best_scores[idx].item())
            if score < self.score_threshold:
                continue

            cls_idx = int(best_classes[idx].item())
            if cls_idx < 0 or cls_idx >= len(vocab):
                continue

            mask_prob = torch.sigmoid(masks_tensor[idx]).numpy()
            mask = mask_prob > self.mask_threshold

            area = int(mask.sum())
            if area <= 0:
                continue

            candidates.append({
                'mask': mask,
                'score': score,
                'label': vocab[cls_idx],
                'area': area,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item['score'], item['area']), reverse=True)
        return candidates[0]

    def _is_reasonable_mask(self, mask: np.ndarray) -> bool:
        if mask is None:
            return False

        h, w = mask.shape[:2]
        total_area = h * w
        mask_area = int(mask.sum())

        if mask_area <= 0:
            return False

        area_ratio = mask_area / float(total_area)
        if area_ratio > self.max_mask_area_ratio:
            return False

        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return False

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        bw = x2 - x1 + 1
        bh = y2 - y1 + 1

        if bw < self.min_bbox_side or bh < self.min_bbox_side:
            return False

        bbox_area_ratio = (bw * bh) / float(total_area)
        if bbox_area_ratio > 0.85:
            return False

        return True

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
        label: str | None = None,
        score: float | None = None,
    ) -> np.ndarray:
        out = image_bgr.copy()
        out[mask] = (0, 255, 0)

        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

            text = label or 'object'
            if score is not None:
                text = f'{text} ({score:.2f})'

            cv2.putText(
                out,
                text,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if center_coords is not None:
            cv2.circle(out, center_coords, 5, (0, 0, 255), -1)

        return out

    @staticmethod
    def get_center_coordinates(mask: np.ndarray) -> tuple[int, int] | None:
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        x_median = float(np.median(x_indices))
        y_median = float(np.median(y_indices))

        points = np.column_stack((x_indices, y_indices)).astype(np.float32)
        target = np.array([x_median, y_median], dtype=np.float32)

        distances_sq = np.sum((points - target) ** 2, axis=1)
        best_idx = int(np.argmin(distances_sq))

        x_center = int(points[best_idx, 0])
        y_center = int(points[best_idx, 1])

        return x_center, y_center