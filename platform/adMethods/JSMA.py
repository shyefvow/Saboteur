import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 计算雅可比矩阵，即前向导数
def compute_jacobian(model, input, device):
    var_input = input.clone()

    var_input.detach_()
    var_input.requires_grad = True
    output = model(var_input)

    num_features = int(np.prod(var_input.shape[1:]))
    jacobian = torch.zeros([output.size()[1], num_features])
    for i in range(output.size()[1]):
        # zero_gradients(input)
        if var_input.grad is not None:
          var_input.grad.zero_()
        # output.backward(mask,retain_graph=True)
        output[0][i].backward(retain_graph=True)
        # copy the derivative to the target place
        jacobian[i] = var_input.grad.squeeze().view(-1, num_features).clone()

    return jacobian.to(device)


# 计算显著图
def saliency_map(jacobian, target_index, increasing, search_space, nb_features, device):
    domain = torch.eq(search_space, 1).float()  # The search domain
    # the sum of all features' derivative with respect to each class
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_index]  # The forward derivative of the target class
    others_grad = all_sum - target_grad  # The sum of forward derivative of other classes

    # this list blanks out those that are not in the search domain
    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float().to(device)
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(device)
    increase_coef = increase_coef.view(-1, nb_features)

    # calculate sum of target forward derivative of any 2 features.
    target_tmp = target_grad.clone()
    target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)  # PyTorch will automatically extend the dimensions
    # calculate sum of other forward derivative of any 2 features.
    others_tmp = others_grad.clone()
    others_tmp += increase_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    # zero out the situation where a feature sums with itself
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte().to(device)

    # According to the definition of saliency map in the paper (formulas 8 and 9),
    # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
    if increasing:
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)
    # apply the mask to the saliency map
    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
    # do the multiplication according to formula 10 in the paper
    saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    # get the most significant two pixels
    max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q


class JSMAAttack:
    def __init__(self, model, device, theta=1.0, gamma=0.1):
        """

        :param model:
        :param device:
        :param theta: 扰动向量中每个像素点的增量或减量
        :param gamma: 最多扰动特征数占总特征数量的比例
        """
        self.model = model.to(device)
        self.device = device
        self.theta = theta
        self.gamma = gamma

    def __call__(self, batch_image, ys_target):
        """

        :param batch_image:
        :param ys_target: 目标标签
        :return:
        """
        adv_samples = []
        for i in range(batch_image.shape[0]):
            image = batch_image[i]
            copy_sample = np.copy(image)

            var_sample = Variable(torch.from_numpy(copy_sample), requires_grad=True).to(self.device)
            var_target = Variable(torch.LongTensor([ys_target, ])).to(self.device)

            if self.theta > 0:
                increasing = True
            else:
                increasing = False

            num_features = int(np.prod(copy_sample.shape[1:]))
            shape = var_sample.size()

            max_iters = int(np.ceil(num_features * self.gamma / 2.0))

            if increasing:
                search_domain = torch.lt(var_sample, 0.99)
            else:
                search_domain = torch.gt(var_sample, 0.01)
            search_domain = search_domain.view(num_features)

            output = self.model(var_sample)
            current = torch.max(output.data, 1)[1].cpu().numpy()

            iter = 0
            while (iter < max_iters) and (current[0] != ys_target) and (search_domain.sum() != 0):
                # calculate Jacobian matrix of forward derivative
                jacobian = compute_jacobian(self.model, var_sample, self.device)
                # get the saliency map and calculate the two pixels that have the greatest influence
                p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features, self.device)
                # apply modifications
                var_sample_flatten = var_sample.view(-1, num_features).clone().detach_()
                var_sample_flatten[0, p1] += self.theta
                var_sample_flatten[0, p2] += self.theta

                new_sample = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
                new_sample = new_sample.view(shape)
                search_domain[p1] = 0
                search_domain[p2] = 0
                var_sample = Variable(torch.tensor(new_sample), requires_grad=True).to(self.device)

                output = self.model(var_sample)
                current = torch.max(output.data, 1)[1].cpu().numpy()
                iter += 1

            adv_samples.append(var_sample.data.cpu().numpy())
        return torch.stack([torch.Tensor(s) for s in adv_samples])


