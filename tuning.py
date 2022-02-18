import cv2

hyperparams = {
    # Clahe filter parameters
    'clahe': {
        'clipLimit': 0,
        'tileGridSize': (1, 1)
    },

    # Frame preprocessing parameters
    'preprocessing': {
        # see more at cv2.bilateralFilter() documentation
        # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
        'd': 5,
        'sigmaColor': 50.0,
        'sigmaSpace': 50.0
    },

    # Background extraction parameters
    'bg_extraction': {
        # How many frames to use for creating the background?
        'frame_count': 60,
        # Take one frame every {frame_skip} frames to compute the background.
        'frame_skip': 1,

        # Blending parameter for mean-based and median-based backgrounds.
        'alpha': 0.5,
        # Filter out blobs that are smaller than 0.1% of the frame size, in terms of area.
        'area_thresh_perc': 0.001
    },

    # Background subtraction parameters
    'bg_subtraction': {
        # Level of intensity to which apply a threshold, in order to obtain a mask.
        'diff_threshold': 40.0,
    },

    # Mask refinement and cleaning parameters
    # See also kernels
    'mask_cleaning': {
        # How much padding to add to the mask before morphological operations
        'padding_size': 32,
    },

    # Entity classification parameters
    'tracking': {
        # Two blobs don't belong to the same entity if they overlap *less* than this number
        'intersection_threshold': 15.0,

        # Two blobs don't belong to the same entity if their aspect differs *more* than this.
        'aspect_threshold': 1.9,

        # Two blobs don't belong to the same entity if their extent differs *more* than this.
        'extent_threshold': 0.25,

        # Two blobs don't belong to the same entity if their solidity differs *more* than this.
        'solidity_threshold': 0.2,
    },
}

kernels = {
    'small': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
    'big': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16)),
    'vertical': (50, 1)
}