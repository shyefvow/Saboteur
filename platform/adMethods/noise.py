import math

import numpy as np
import torch
import foolbox as fb

class GaussianNoiseAttack():
    """Add Gaussian noise to the input."""

    def __init__(self, mean=0.0, std=0.1):
        """
        :param mean: Mean of the Gaussian noise.
        :param std: Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, input, label=None, unpack=True):
        """Perform the Gaussian noise attack.

        :param input: The input to perturb.
        :param label: The true label of the input. This parameter is unused.
        :param unpack: If True, returns a tuple (noise, perturbed_input).
            Otherwise, returns the perturbed input.
        :return: The perturbed input or a tuple (noise, perturbed_input).
        """

        noise = np.random.normal(self.mean, self.std, input.shape)
        perturbed_input = input.cpu() + noise

        if unpack:
            return noise, perturbed_input
        else:
            return perturbed_input


class pepperAttack():
    """Add Pepper noise to the input."""

    def __init__(self, prob=10):
        """
        :param prob:
        """
        self.prob = prob

    def __call__(self, input, label=None):
        """Perform the Pepper noise attack.

        :param input: The input to perturb.
        :param label: The true label of the input. This parameter is unused.
        :return: The perturbed input or a tuple (noise, perturbed_input).
        """

        assert len(input.size()) == 4, "Input tensor shape should be (B, C, H, W)"

        B, c, w, h = input.shape
        self.prob /= 100

        for i in range(B):
            mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[1 - self.prob, self.prob / 2., self.prob / 2.])
            mask = torch.tensor(np.repeat(mask, c, axis=0))  # 按channel 复制到 与img具有相同的shape

            max = torch.max(input)
            min = torch.min(input)

            input[i][mask == 1] = max
            input[i][mask == 2] = min

        return input


class foggyAttack():
    def __init__(self, alpha=0.5, beta=0.01):
        """

        :param alpha: 亮度
        :param beta: 雾的浓度
        """
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input):
        """
        :param input: torch.Tensor, 输入图像数据，大小为 (B, C, H, W)
        :return: torch.Tensor, 雾化扰动之后的数据，大小为 (B, C, H, W)
        """

        assert len(input.size()) == 4, "Input tensor shape should be (B, C, H, W)"

        B, C, H, W = input.shape

        input /= 255

        size = math.sqrt(max(H, W))     # 雾化尺寸
        center = (H // 2, W // 2)       # 雾化中心
        for k in range(H):
            for l in range(W):
                d = -0.04 * math.sqrt((k - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-self.beta * d)
                input[:, :, k, l] = input[:, :, k, l] * td + self.alpha * (1 - td)


        input *= 255
        return input