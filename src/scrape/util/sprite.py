from functools import cached_property
from os import path

import imageio.v3 as iio
import requests
import numpy as np


class Sprite:
    def __init__(self, img: np.ndarray):
        height, width, channels = img.shape
        assert height == 42
        assert width in (52, 56)
        assert channels == 4

        self.img = img / 255.  # normalize the image
        self.height = height
        self.width = width

    @staticmethod
    def fetch(url: str, cache_path: str) -> "Sprite":
        if not path.exists(cache_path):
            req = requests.get(url)
            with open(cache_path, "wb") as f:
                f.write(req.content)
        img = iio.imread(cache_path, extension=".png", mode="RGBA")
        return Sprite(img)

    @property
    def red(self) -> np.ndarray:
        return self.img[:, :, 0]

    @property
    def green(self) -> np.ndarray:
        return self.img[:, :, 1]

    @property
    def blue(self) -> np.ndarray:
        return self.img[:, :, 2]

    @cached_property
    def brightness(self) -> np.ndarray:
        return (self.red + self.green + self.blue) / 3.

    @property
    def alpha(self) -> np.ndarray:
        return self.img[:, :, 3]

    @cached_property
    def perimeter(self) -> np.ndarray:
        is_perimeter = np.zeros_like(self.alpha)
        for i in range(self.height):
            for j in range(self.width):
                if self.alpha[i, j] > 0.:
                    if (
                        i == 0 or i == self.height - 1
                        or j == 0 or j == self.width - 1
                    ):
                        is_perimeter[i, j] = 1.
                    elif (
                        self.alpha[i-1, j] == 0.
                        or self.alpha[i+1, j] == 0.
                        or self.alpha[i, j-1] == 0.
                        or self.alpha[i, j+1] == 0.
                    ):
                        is_perimeter[i, j] = 1.
        return is_perimeter
