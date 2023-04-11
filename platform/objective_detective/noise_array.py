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
