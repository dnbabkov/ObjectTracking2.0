import gc
import os
import time
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from ament_index_python.packages import get_package_share_directory
from PIL import Image as PILImage
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from object_tracking_2.segmentators.base_segmentator import BaseSegmentator, SegmentationResult


class DinoSAMSegmentator(BaseSegmentator):
    def __init__(
        self,
        dino_model_name: str = 'IDEA-Research/grounding-dino-tiny',
        sam_model_type: str = 'vit_h',
        sam_checkpoint_name: str = 'sam_vit_h_4b8939.pth',
        dino_box_threshold: float = 0.4,
        dino_text_threshold: float = 0.3,
        selection_threshold: float = 0.75,
        min_mask_area: int = 200,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)

        self.use_amp = self.device == 'cuda'
        self.amp_dtype = torch.float16

        self.dino_box_threshold = dino_box_threshold
        self.dino_text_threshold = dino_text_threshold
        self.selection_threshold = selection_threshold
        self.min_mask_area = min_mask_area

        share_dir = get_package_share_directory('object_tracking_2')
        checkpoint_path = os.path.join(
            share_dir,
            'model_weights',
            sam_checkpoint_name,
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'SAM checkpoint not found: {checkpoint_path}'
            )

        self.sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)
        self.sam.eval()
        self.sam.requires_grad_(False)
        self.predictor = SamPredictor(self.sam)

        self.dino_processor = AutoProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_name
        )
        self.dino_model.to(self.device)
        self.dino_model.eval()
        self.dino_model.requires_grad_(False)

    @property
    def name(self) -> str:
        return 'DinoSAM'

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()

    def _clear_cuda_memory(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)
        image_pil = PILImage.fromarray(image_rgb)

        text = str(prompt).strip().lower()
        if not text.endswith('.'):
            text += '.'

        total_start_time = time.perf_counter()
        dino_start_time = time.perf_counter()

        try:
            with torch.inference_mode():
                with self._autocast_context():
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
            )

            dino_time = time.perf_counter() - dino_start_time
            result = results[0]

            filtered = [
                (box.detach().cpu().numpy(), float(score.item()), label)
                for box, score, label in zip(
                    result['boxes'],
                    result['scores'],
                    result['labels'],
                )
                if float(score.item()) >= self.selection_threshold
            ]

            del outputs
            del inputs
            del results
            del result
            self._clear_cuda_memory()

            if not filtered:
                total_time = time.perf_counter() - total_start_time
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=total_time,
                    metadata={
                        'dino_time': dino_time,
                        'sam_time': 0.0,
                        'detections_found': 0,
                    },
                )

            box, score, label = max(filtered, key=lambda item: item[1])

            sam_start_time = time.perf_counter()

            with torch.inference_mode():
                with self._autocast_context():
                    self.predictor.set_image(image_rgb)

                    input_boxes = torch.tensor(
                        [box],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    transformed_boxes = self.predictor.transform.apply_boxes_torch(
                        input_boxes,
                        image_rgb.shape[:2],
                    )

                    masks, _, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

            sam_time = time.perf_counter() - sam_start_time
            total_time = time.perf_counter() - total_start_time

            mask = masks[0, 0].detach().cpu().numpy().astype(bool)
            mask_area = int(np.sum(mask))

            del masks
            del input_boxes
            del transformed_boxes
            self.predictor.reset_image()
            self._clear_cuda_memory()

            if mask_area < self.min_mask_area:
                return SegmentationResult(
                    vis_image=image.copy(),
                    center_coords=None,
                    segmentation_time=total_time,
                    metadata={
                        'dino_time': dino_time,
                        'sam_time': sam_time,
                        'mask_area': mask_area,
                        'score': score,
                        'label': label,
                    },
                )

            center_coords = self.get_center_coordinates(mask)

            image_out = image.copy()
            image_out[mask] = (0, 255, 0)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image_out,
                f'{label} ({score:.2f})',
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

            return SegmentationResult(
                vis_image=image_out,
                center_coords=center_coords,
                segmentation_time=total_time,
                metadata={
                    'dino_time': dino_time,
                    'sam_time': sam_time,
                    'mask_area': mask_area,
                    'score': score,
                    'label': label,
                },
            )

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                try:
                    self.predictor.reset_image()
                except Exception:
                    pass
                self._clear_cuda_memory()
            raise

    @staticmethod
    def get_center_coordinates(mask: np.ndarray) -> tuple[int, int] | None:
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))

        return x_mean, y_mean