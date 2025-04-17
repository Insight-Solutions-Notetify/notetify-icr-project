
class SegmentationConfig:
    MIN_CHAR_SIZE = (10, 5) # Pixels may be larger or smaller than expected
    MAX_VALUE = 255

    # Line Segmentation
    HEIGHT_CHAR_BUFFER = 10
    LINE_GAP_FACTOR = 0.5
    TEXT_LINE_THRESHOLD = 0.1

    # Word Segmentation
    WIDTH_CHAR_BUFFER = 10
    WORD_GAP_FACTOR = 1.8
    MIN_WORD_IMG_HEIGHT = 30
    
    # Character Segmentation
    PROXIMITY_THRESHOLD=15

    MINIMUM_SPLIT_WIDTH = 1.6 # Ignore splitting if average width goes below this value
    HEIGHT_INFLUENCE = 0.25

    # Post processing
    IMAGE_DIMS = (28, 28)

segmentation_config = SegmentationConfig()