"""This module apply Sliding Window technique for an image object."""


def sliding_window(source, w_size: tuple, stride: int):
    """
    A Sliding Window generator that yields window patches from original high-resolution input image.

    :param source: (numpy array) input image.
    :param w_size:  size of the sliding window, as (width, height), in pixels.
    :param stride:  stride for sliding the window, in pixels.

    Yields:
    numpy array: small w_size window patch from original image.
    """

    image_h, image_w = source.shape[:2]
    window_w, window_h = w_size

    # Slide the window over the image
    for y in range(0, image_h - window_h + 1, stride):
        for x in range(0, image_w - window_w + 1, stride):
            yield x, y, source[y:y + window_h, x:x + window_w]
