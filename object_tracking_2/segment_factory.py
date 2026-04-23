#from object_tracking_2.segmentators.clip_segmentator import CLIPSegmentator
#from object_tracking_2.segmentators.dino_sam_segmentator import DinoSAMSegmentator
#from object_tracking_2.segmentators.openseed_segmentator import OpenSeeDSegmentator
#from object_tracking_2.segmentators.seem_segmentator import SEEMSegmentator
#from object_tracking_2.segmentators.sam2_segmentator import SAM2Segmentator
#from object_tracking_2.segmentators.dinov2_segmentator import DINOv2Segmentator

def create_segmentator(name: str):
    n = name.strip().lower()

    if n == 'clip':
        from object_tracking_2.segmentators.clip_segmentator import CLIPSegmentator
        return CLIPSegmentator()

    if n in ('dinosam', 'dino_sam', 'groundingsam', 'grounding_sam'):
        from object_tracking_2.segmentators.dino_sam_segmentator import DinoSAMSegmentator
        return DinoSAMSegmentator()

    if n == 'seem':
        from object_tracking_2.segmentators.seem_segmentator import SEEMSegmentator
        return SEEMSegmentator()

    if n == 'openseed':
        from object_tracking_2.segmentators.openseed_segmentator import OpenSeeDSegmentator
        return OpenSeeDSegmentator()

    if n == 'sam2':
        from object_tracking_2.segmentators.sam2_segmentator import SAM2Segmentator
        return SAM2Segmentator()

    if n == 'dinov2':
        from object_tracking_2.segmentators.dinov2_segmentator import DINOv2Segmentator
        return DINOv2Segmentator()

    raise ValueError(f'Unknown segmentator: {name}')