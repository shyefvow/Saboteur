import math

import numpy as np


class GaussianNoiseAttack():
    """Add Gaussian noise to the input."""

    def __init__(self, mean=0.0, std=0.1):
        """
        :param mean: Mean of the Gaussian noise.
        :param std: Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std
        self.min_value = None
        self.max_value = None

    def __call__(self, input, label=None, unpack=True):
        """Perform the Gaussian noise attack.

        :param input: The input to perturb. The shape can be arbitrary.
        :param label: The true label of the input. This parameter is unused.
        :param unpack: If True, returns a tuple (noise, perturbed_input).
            Otherwise, returns the perturbed input.
        :return: The perturbed input or a tuple (noise, perturbed_input).
        """

        if self.min_value is None or self.max_value is None:
            self.min_value = np.min(input)
            self.max_value = np.max(input)

        input_normalized = (input - self.min_value) / (self.max_value - self.min_value)
        noise = np.random.normal(self.mean, self.std, input.shape)
        perturbed_input_normalized = input_normalized + noise
        perturbed_input_normalized = np.clip(perturbed_input_normalized, 0.0, 1.0)
        perturbed_input = perturbed_input_normalized * (self.max_value - self.min_value) + self.min_value

        if unpack:
            return noise, perturbed_input
        else:
            return perturbed_input


class pepperAttack():
    def __init__(self, prob=0.05):
        """
        :param prob: The probability of adding salt and pepper noise.
        """
        self.prob = prob

    def __call__(self, input, label=None):
        """Perform the salt and pepper noise attack.

        :param input: The input array to perturb. The shape can be arbitrary.
        :param label: The true label of the input. This parameter is unused.
        :return: The perturbed input.
        """

        assert isinstance(input, np.ndarray), "Input should be a numpy array."

        noise = np.random.choice([0, 1, 2], size=input.shape, p=[1 - self.prob, self.prob / 2., self.prob / 2.])
        max_value = np.max(input)
        min_value = np.min(input)

        perturbed_input = input.copy()
        perturbed_input[noise == 1] = max_value
        perturbed_input[noise == 2] = min_value

        return perturbed_input


class foggyAttack():
    def __init__(self, alpha=0.5, beta=0.01):
        """
        :param alpha: Brightness.
        :param beta: Fog density.
        """
        self.alpha = alpha
        self.beta = beta
    def __call__(self, input):
        """
        :param input: numpy.ndarray, input image data with shape (H, W, C)
        :return: numpy.ndarray, foggy perturbed data with shape (H, W, C)
        """

        assert isinstance(input, np.ndarray), "Input should be a numpy array."
        assert len(input.shape) == 3, "Input array shape should be (H, W, C)"

        H, W, C = input.shape

        input = input.astype(np.float32) / 255

        size = math.sqrt(max(H, W))  # Fog size
        center = (H // 2, W // 2)  # Fog center
        for k in range(H):
            for l in range(W):
                d = -0.04 * math.sqrt((k - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-self.beta * d)
                input[k, l, :] = input[k, l, :] * td + self.alpha * (1 - td)

        input = (input * 255).astype(np.uint8)
        return input
