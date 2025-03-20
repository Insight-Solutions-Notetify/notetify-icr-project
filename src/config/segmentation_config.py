
class SegmentationConfig:
    MIN_CHAR_SIZE = (5, 15)
    MIN_WORD_GAP = 5
    MIN_LINE_GAP = 5


    # Character segmentation
    GAP_THRESHOLD = 0.01
    MIN_PIXEL_DIFF = 2
    MIN_CHAR_WIDTH = 5 * 0.8 # 80% of the average character width

segmentation_config = SegmentationConfig()