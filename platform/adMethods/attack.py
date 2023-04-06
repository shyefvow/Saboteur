import torch

class FGMAttack:
    def __init__(self, model, epsilon=0.05):
        """
        :param model: 要攻击的 PyTorch 模型
        :param epsilon: 扰动大小
        """
        self.model = model
        self.epsilon = epsilon

    def __call__(self, inputs, targets):
        """
        对给定的输入和目标进行攻击，返回扰动后的输入
        :param inputs: torch.Tensor, 输入图像数据，大小为 (B, C, H, W)
        :param targets: torch.Tensor, 目标标签，大小为 (B, )
        :return: torch.Tensor, 扰动后的输入图像，大小为 (B, C, H, W)
        """

        inputs.requires_grad = True
        inputs = inputs.cuda()
        targets = targets.cuda()
        # 进行前向传播
        outputs = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        # 对输入进行反向传播
        loss.backward()

        # 计算扰动并添加到输入上
        inputs_grad = torch.sign(inputs.grad)
        inputs_adv = inputs + self.epsilon * inputs_grad

        return inputs_adv


class FGSMAttack:
    def __init__(self, model, epsilon=0.05):
        """
        :param model: torch.nn.Module, 要攻击的模型
        :param epsilon: float, 攻击强度
        """
        self.model = model
        self.epsilon = epsilon

    def __call__(self, inputs, targets):
        """
        :param inputs: torch.Tensor, 输入图像数据，大小为 (B, C, H, W)
        :param targets: torch.Tensor, 目标标签数据，大小为 (B,)
        :return: torch.Tensor, 攻击后的图像数据，大小为 (B, C, H, W)
        """
        inputs.requires_grad = True  # 开启梯度计算
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)  # 计算交叉熵损失

        gradients = torch.autograd.grad(loss, inputs)[0]  # 计算梯度
        sign_gradients = gradients.sign()  # 取梯度的符号

        # 对原始图像进行扰动
        adv_inputs = inputs + self.epsilon * sign_gradients
        adv_inputs = torch.clamp(adv_inputs, min=0, max=1)  # 把像素值限制在0~1范围内

        return adv_inputs


class PGDAttack:
    def __init__(self, model, epsilon=0.1, alpha=0.01, num_steps=10, random_start=True):
        """
        :param model: torch.nn.Module, 要攻击的模型
        :param epsilon: float, 攻击强度
        :param alpha: float, 梯度步长
        :param num_steps: int, 迭代步数
        :param random_start: bool, 是否从随机起点开始攻击
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start

    def __call__(self, inputs, targets):
        """
        :param inputs: torch.Tensor, 输入图像数据，大小为 (B, C, H, W)
        :param targets: torch.Tensor, 目标标签数据，大小为 (B,)
        :return: torch.Tensor, 攻击后的图像数据，大小为 (B, C, H, W)
        """
        inputs = inputs.cuda()
        targets = targets.cuda()

        if self.random_start:
            adv_inputs = inputs + torch.empty_like(inputs).uniform_(-self.epsilon, self.epsilon)  # 从随机起点开始攻击
            adv_inputs = torch.clamp(adv_inputs, min=0, max=1)  # 把像素值限制在0~1范围内
        else:
            adv_inputs = inputs.clone()

        for _ in range(self.num_steps):
            adv_inputs.detach_()
            adv_inputs.requires_grad = True  # 开启梯度计算
            outputs = self.model(adv_inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)  # 计算交叉熵损失

            gradients = torch.autograd.grad(loss, adv_inputs)[0]  # 计算梯度
            sign_gradients = gradients.sign()  # 取梯度的符号

            # 对图像进行扰动
            adv_inputs = adv_inputs + self.alpha * sign_gradients
            adv_inputs = torch.max(torch.min(adv_inputs, inputs + self.epsilon), inputs - self.epsilon)  # 投影到 L-infinity 范围内
            adv_inputs = torch.clamp(adv_inputs, min=0, max=1)  # 把像素值限制在0~1范围内
        return adv_inputs


class CW_untarget:
    def __init__(self, model, confidence=0, max_iterations=1000, binary_search_steps=10, learning_rate=0.05,
                 initial_const=0.001, clip_min=0.0, clip_max=1.0):
        """
        :param model: torch.nn.Module, 要攻击的模型
        :param confidence: float, 用于计算置信度的参数，取值越大，攻击越保守
        :param max_iterations: int, 最大迭代步数
        :param binary_search_steps: int, 二分搜索步数
        :param learning_rate: float, 学习率
        :param initial_const: float, 初始扰动大小
        :param clip_min: float, 输入图像的最小值
        :param clip_max: float, 输入图像的最大值
        """
        self.model = model
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.learning_rate = learning_rate
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, inputs):
        """
        :param inputs: torch.Tensor, 输入图像数据，大小为 (B, C, H, W)
        :return: torch.Tensor, 攻击后的图像数据，大小为 (B, C, H, W)
        """
        batch_size = inputs.size(0)
        dtype = inputs.dtype
        device = inputs.device

        # 初始化扰动
        adv_inputs = inputs + torch.empty_like(inputs).uniform_(-self.initial_const, self.initial_const)
        adv_inputs = torch.clamp(adv_inputs, self.clip_min, self.clip_max)

        # 迭代优化
        for i in range(self.max_iterations):
            adv_inputs.detach_()
            adv_inputs.requires_grad = True  # 开启梯度计算
            outputs = self.model(adv_inputs)
            logits = torch.log_softmax(outputs, dim=1)
            correct_logprobs = logits.gather(1, outputs.max(1)[1].view(-1, 1)).squeeze()

            # 计算置信度
            if self.confidence > 0:
                target_logprobs = torch.clamp(logits - correct_logprobs.view(-1, 1), min=-self.confidence)
            else:
                target_logprobs = logits - correct_logprobs.view(-1, 1)

            # 损失函数：maximize(target_logprobs)
            loss = -torch.sum(target_logprobs)

            # 计算梯度
            gradients = torch.autograd.grad(loss, adv_inputs)[0]

            # 二分搜索找到最小扰动
            for binary_search_step in range(self.binary_search_steps):
                # 用学习率更新扰动
                adv_inputs_new = adv_inputs.detach() + self.learning_rate * gradients.sign()
                adv_inputs_new = torch.clamp(adv_inputs_new, self.clip_min, self.clip_max)

                # 更新扰动并重新计算梯度
                adv_inputs_new.detach_()
                adv_inputs_new.requires_grad = True
                outputs_new = self.model(adv_inputs_new)
                logits_new = torch.log_softmax(outputs_new, dim=1)

                correct_logprobs_new = logits_new.gather(1, outputs_new.max(1)[1].view(-1, 1)).squeeze()
                if self.confidence > 0:
                    target_logprobs_new = torch.clamp(logits_new - correct_logprobs_new.view(-1, 1),
                                                      min=-self.confidence)
                else:
                    target_logprobs_new = logits_new - correct_logprobs_new.view(-1, 1)

                loss_new = -torch.sum(target_logprobs_new)

                if loss_new < loss:
                    # 更新最小扰动
                    loss = loss_new
                    adv_inputs = adv_inputs_new

            # 重新限制扰动范围
            adv_inputs = torch.clamp(adv_inputs, self.clip_min, self.clip_max)

        return adv_inputs
