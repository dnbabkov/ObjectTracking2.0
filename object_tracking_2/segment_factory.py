from object_tracking_2.segmentators.clip_segmentator import CLIPSegmentator
from object_tracking_2.segmentators.dino_sam_segmentator import DinoSAMSegmentator
from object_tracking_2.segmentators.openseed_segmentator import OpenSeeDSegmentator
from object_tracking_2.segmentators.seem_segmentator import SEEMSegmentator


def create_segmentator(name: str):
    segmentators = {
        'CLIP': CLIPSegmentator,
        'DinoSAM': DinoSAMSegmentator,
        'OpenSeeD': OpenSeeDSegmentator,
        'SEEM': SEEMSegmentator,
    }

    if name not in segmentators:
        available = ', '.join(segmentators.keys())
        raise ValueError(
            f'Unsupported segmentator "{name}". Available: {available}'
        )

    return segmentators[name]()