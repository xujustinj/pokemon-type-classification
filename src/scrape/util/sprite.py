from functools import cached_property
from os import path

import imageio.v3 as iio
import requests
import numpy as np


class Sprite:
    """A low-resolution image of a Pokemon.

    Args:
        img (ndarray): [H x W x C] A 3-D array of RGBA values.
    """
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
        """Retrieves a sprite image from the given URL.

        Args:
            url (str): The URL of the image.
            cache_path (str): The path to a local file to cache the image at.

        Returns:
            sprite (Sprite): The sprite.
        """
        if not path.exists(cache_path):
            req = requests.get(url)
            with open(cache_path, "wb") as f:
                f.write(req.content)
        img = iio.imread(cache_path, extension=".png", mode="RGBA")
        return Sprite(img)

    @property
    def red(self) -> np.ndarray:
        """The red channel of the sprite.

        Returns:
            red (ndarray): [H x W] The red value of each pixel, from 0 to 1.
        """
        return self.img[:, :, 0]

    @property
    def green(self) -> np.ndarray:
        """The green channel of the sprite.

        Returns:
            green (ndarray): [H x W] The green value of each pixel, from 0 to 1.
        """
        return self.img[:, :, 1]

    @property
    def blue(self) -> np.ndarray:
        """The blue channel of the sprite.

        Returns:
            blue (ndarray): [H x W] The blue value of each pixel, from 0 to 1.
        """
        return self.img[:, :, 2]

    @cached_property
    def brightness(self) -> np.ndarray:
        """The brightness channel of the sprite.

        Returns:
            brightness (ndarray): [H x W] The brightness of each pixel, from 0
                to 1.
        """
        return (self.red + self.green + self.blue) / 3.

    @property
    def alpha(self) -> np.ndarray:
        """The opacity channel of the sprite.

        Returns:
            alpha (ndarray): [H x W] The opacity of each pixel, from 0 to 1.
        """
        return self.img[:, :, 3]

    @cached_property
    def perimeter(self) -> np.ndarray:
        """The perimeter channel of the sprite.

        A perimeter pixel is any opaque pixel that is either on the border of
        the sprite, or orthogonally adjacent to a transparent pixel.

        Returns:
            perimeter (ndarray): [H x W] 1 if the pixel is on the perimeter or
                0 otherwise.
        """
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
