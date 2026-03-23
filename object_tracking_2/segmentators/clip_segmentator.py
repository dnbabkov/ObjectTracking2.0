import time

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torch.nn.functional import interpolate
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from object_tracking_2.segmentators.base_segmentator import BaseSegmentator, SegmentationResult


class CLIPSegmentator(BaseSegmentator):
    def __init__(
        self,
        model_name: str = 'CIDAS/clipseg-rd64-refined',
        threshold: float = 0.85,
        min_mask_area: int = 200,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.threshold = threshold
        self.min_mask_area = min_mask_area

        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return 'CLIP'

    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:

        print("seg_in")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)

        start_time = time.perf_counter()

        inputs = self.processor(
            text=[prompt],
            images=image_pil,
            return_tensors='pt',
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits

        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        upsampled_logits = interpolate(
            logits,
            size=image_pil.size[::-1],
            mode='bilinear',
            align_corners=False,
        )

        probabilities = upsampled_logits.sigmoid()[0, 0].detach().cpu().numpy()
        mask = probabilities > self.threshold

        segmentation_time = time.perf_counter() - start_time

        mask_area = int(np.sum(mask))
        if mask_area < self.min_mask_area:
            return SegmentationResult(
                vis_image=image.copy(),
                center_coords=None,
                segmentation_time=segmentation_time,
                metadata={
                    'mask_area': mask_area,
                    'threshold': self.threshold,
                },
            )

        center_coords = self.get_center_coordinates(mask)

        image_out = image.copy()
        image_out[mask] = (0, 255, 0)

        return SegmentationResult(
            vis_image=image_out,
            center_coords=center_coords,
            segmentation_time=segmentation_time,
            metadata={
                'mask_area': mask_area,
                'threshold': self.threshold,
            },
        )

    @staticmethod
    def get_center_coordinates(mask: np.ndarray) -> tuple[int, int] | None:
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))

        return x_mean, y_mean