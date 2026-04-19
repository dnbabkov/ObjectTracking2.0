from object_tracking_2.segmentators.clip_segmentator import CLIPSegmentator
from object_tracking_2.segmentators.dino_sam_segmentator import DinoSAMSegmentator
from object_tracking_2.segmentators.openseed_segmentator import OpenSeeDSegmentator
from object_tracking_2.segmentators.seem_segmentator import SEEMSegmentator
from object_tracking_2.segmentators.sam2_segmentator import SAM2Segmentator
from object_tracking_2.segmentators.dinov2_segmentator import DINOv2Segmentator

def create_segmentator(name: str):
    segmentators = {
        'CLIP':     CLIPSegmentator,
        'DinoSAM':  DinoSAMSegmentator,
        'OpenSeeD': OpenSeeDSegmentator,
        'SEEM':     SEEMSegmentator,
        'SAM2':     SAM2Segmentator,
        'DINOv2':   DINOv2Segmentator,
    }

    if name not in segmentators:
        available = ', '.join(segmentators.keys())
        raise ValueError(
            f'Unsupported segmentator "{name}". Available: {available}'
        )

    return segmentators[name]()