from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SegmentationResult:
    vis_image: np.ndarray
    center_coords: tuple[int, int] | None
    segmentation_time: float
    depth_map: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSegmentator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def segment(self, image, prompt: str, depth=None) -> SegmentationResult:
        pass