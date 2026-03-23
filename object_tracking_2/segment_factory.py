from object_tracking_2.segmentators.clip_segmentator import CLIPSegmentator
from object_tracking_2.segmentators.dino_sam_segmentator import DinoSAMSegmentator


def create_segmentator(name: str):
    segmentators = {
        'CLIP': CLIPSegmentator,
        'DinoSAM': DinoSAMSegmentator,
    }

    if name not in segmentators:
        available = ', '.join(segmentators.keys())
        raise ValueError(
            f'Unsupported segmentator "{name}". Available: {available}'
        )

    return segmentators[name]()