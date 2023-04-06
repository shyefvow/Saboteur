import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import foolbox as fb
import torch
import torch.nn as nn
from model import Net
from foolbox.criteria import TargetedMisclassification

import torchvision
from torchvision import datasets, transforms

from adMethods.noise import *
from adMethods.attack import *

batch_size = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = nn.CrossEntropyLoss()

# 读取模型
# model = torchvision.models.resnet50(pretrained=True)
model = Net().to(device)
model.load_state_dict(torch.load("model/model.pt"))
model.eval()

# # 通过 fb.PyTorchModel 封装成的类，其 fmodel 使用与我们训练的 simple_model 基本一样
# fmodel = fb.PyTorchModel(model, bounds=(0, 1))
#
#
# # 如下代码 dataset 可以选择 cifar10,cifar100,imagenet,mnist 等。图像格式是 channel_first
# # 由于 fmodel 设置了 bounds，如下代码 fb.utils.samples 获得(一个 batch)数据，会自动将其转化为 bounds 范围中
# images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=20)
# print(images.shape)
# print(labels.shape)
#
#
# # attack = foggyAttack()
# # attack = GaussianNoiseAttack()
# # attack = foggyAttack()
# attack = CW_untarget(fmodel)
#
# advs = attack(images)
#
# # tensor_np = np.transpose(images.cpu().detach().numpy(), (0, 2, 3, 1))
# # fig, axs = plt.subplots(5, 1, figsize=(5, 20))
# # for i in range(5):
# #     axs[i].imshow(tensor_np[i])
# #     axs[i].axis('off')
# # plt.show()
#
# tensor_np = np.transpose(advs.cpu().detach().numpy(), (0, 2, 3, 1))
# fig, axs = plt.subplots(5, 1, figsize=(5, 20))
# for i in range(5):
#     axs[i].imshow(tensor_np[i])
#     axs[i].axis('off')
# plt.show()
#
# # 准确率测试
# # 该函数就是模型的 test，将 images 与 labels 输入就能判断，模型正确分类的准确率。
# acc = fb.utils.accuracy(fmodel, advs, labels)
# print(acc)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), ])),
    batch_size=batch_size, shuffle=False )

iter_times = 3
plt.figure(figsize=(24, 2.5*iter_times))
cnt = 0
for i, (data, target) in enumerate(test_loader):
    attack = PGDAttack(model)
    data = attack(data, target)
    for j in range(batch_size):
        cnt += 1
        plt.subplot(iter_times, batch_size,cnt)
        if j == 0:
            plt.ylabel('{} batch'.format(i+1), fontsize=12)

        plt.imshow(data.detach().cpu().numpy()[j, :, :, :].reshape(28, 28), cmap='gray')

        predict = model(data.to(device))
        m = nn.Softmax()
        soft_output = m(predict)
        plt.title('{} -> {}'.format(target[j], soft_output.argmax(axis=1)[j]))
    if (i >= iter_times-1):
        break
plt.show()