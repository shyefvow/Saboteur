import copy

import numpy as np
import torch


class DeepFoolAttack:
    def __init__(self, model, num_classes=10, overshoot=0.02, max_iter=100):
        """

        :param model:
        :param num_classes:
        :param overshoot: 单次扰动强度
        :param max_iter: 最多迭代次数
        """
        self.model = model
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter

    def __call__(self, image):
        image = image.cuda()
        image.requires_grad = True

        with torch.no_grad():
            logits = self.model(image)
        f_image = logits.detach().cpu().numpy()
        # 预测分数从高到低排序后的类别索引
        I = np.argsort(f_image, axis=1)[:, ::-1]
        I = I[:, :self.num_classes]
        label = I[:, 0]

        input_shape = image.shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        x = pert_image
        x.requires_grad = True
        fs = self.model.forward(x)
        k_i = label

        # 预测结果正确就继续迭代
        while loop_i < self.max_iter:

            pert = np.inf
            fs[range(len(label)), label].backward(torch.ones(len(label)).cuda(), retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, self.num_classes):
                if x.grad is not None:
                    x.grad.zero_()

                fs[range(len(label)), I[:, k]].backward(torch.ones(len(label)).cuda(), retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[range(len(label)), I[:, k]] - fs[range(len(label)), label]).data.cpu().numpy()

                pert_k = np.abs(f_k) / np.linalg.norm(w_k.reshape(len(label), -1), axis=1)

                # determine which w_k to use
                index_pert_k = np.argmin(pert_k)
                if pert_k[index_pert_k] < pert:
                    pert = pert_k[index_pert_k]
                    w = w_k[index_pert_k]

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = image + (1 + self.overshoot) * torch.from_numpy(r_tot).cuda()

            x = pert_image.detach()
            x.requires_grad = True
            fs = self.model.forward(x)
            k_i = np.argmax(fs.data.cpu().numpy(), axis=1)

            # 预测结果不正确，就退出迭代
            mask = k_i == label
            mask_float = mask.astype(float)
            mask_sum = mask_float.sum()
            if mask_sum == 0:
                break

            loop_i += 1

        return pert_image
