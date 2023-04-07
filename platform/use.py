import warnings
warnings.filterwarnings('ignore')

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
from adMethods.JSMA import *

batch_size = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = nn.CrossEntropyLoss()

# 读取模型
# model = torchvision.models.resnet50(pretrained=True)
model = Net().to(device)
model.load_state_dict(torch.load("model/model.pt"))
model.eval()


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), ])),
    batch_size=batch_size, shuffle=False)

# 画图
iter_times = 3
plt.figure(figsize=(24, 2.5*iter_times))
cnt = 0
for i, (data, target) in enumerate(test_loader):
    attack = JSMAAttack(model, device)
    data = attack(data, 2)

    for j in range(batch_size):
        cnt += 1
        plt.subplot(iter_times, batch_size, cnt)
        # batch 开始
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


# 计算准确率
correct = 0
total = 0
for i, (data, target) in enumerate(test_loader):
    attack = JSMAAttack(model, device)
    data = attack(data, 2)
    predict = model(data.to(device))
    _, predicted = torch.max(predict.data, 1)
    total += target.size(0)
    correct += (predicted == target.to(device)).sum().item()
acc = correct/total
print(acc)
