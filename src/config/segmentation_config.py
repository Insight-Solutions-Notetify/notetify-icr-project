
class SegmentationConfig:
    MIN_CHAR_SIZE = (10, 5)
    MIN_WORD_GAP = 5
    MIN_LINE_GAP = 5

    # Character Segmentation
    MERGING_MIN_X = 3
    MERGING_MIN_Y = 3
    HEIGHT_INFLUENCE = 0.25
    WIDTH_CHAR_BUFFER = 5

    # Post processing
    IMAGE_DIMS = (28, 28)

segmentation_config = SegmentationConfig()