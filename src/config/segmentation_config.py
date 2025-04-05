
class SegmentationConfig:
    MIN_CHAR_SIZE = (10, 5) # Pixels may be larger or smaller than expected
    MAX_VALUE = 255
    MIN_WORD_GAP = 5
    MIN_LINE_GAP = 4

    # Character Segmentation
    MERGING_MIN_X = 3
    MERGING_MIN_Y = 10

    MINIMUM_SPLIT_WIDTH = 1.6 # Ignore splitting if average width goes below this value
    HEIGHT_INFLUENCE = 0.25
    WIDTH_CHAR_BUFFER = 5

    # Post processing
    IMAGE_DIMS = (28, 28)

segmentation_config = SegmentationConfig()