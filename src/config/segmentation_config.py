
class SegmentationConfig:
    MIN_CHAR_SIZE = (10, 5) # Pixels may be larger or smaller than expected
    MAX_VALUE = 255
    MIN_WORD_GAP = 5
    MIN_LINE_GAP = 4

    # Line Segmentation
    HEIGHT_CHAR_BUFFER = 10
    LINE_GAP_FACTOR = 0.5
    TEXT_LINE_THRESHOLD = 0.05

    # Word Segmentation
    WIDTH_CHAR_BUFFER = 10
    WORD_GAP_FACTOR = 1.8
    
    # Character Segmentation
    MERGING_MIN_X = 3
    MERGING_MIN_Y = 10

    MINIMUM_SPLIT_WIDTH = 1.6 # Ignore splitting if average width goes below this value
    HEIGHT_INFLUENCE = 0.25

    # Post processing
    IMAGE_DIMS = (28, 28)

segmentation_config = SegmentationConfig()