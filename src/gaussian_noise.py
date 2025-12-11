import torch
x = torch.randn(3, 64, 64) # ダミーデータ
T = 1000
betas = torch.linspace(0.0001, 0.02, T)

for t in range(T):
  beta = betas[t]
  eps = torch.randn_like(x) # xと同じ形状のガウスノイズを生成
  x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'flower.png')
image = plt.imread(file_path)
print(image.shape)

# 画像の前処理を定義
preprocess = transforms.ToTensor()
x = preprocess(image)
print(x.shape)

def reverse_to_img(x):
  x = x * 255
  x = x.clamp(0, 255)
  x = x.to(torch.uint8)
  to_pil = transforms.ToPILImage()
  return to_pil()

T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
imgs = []

for t in range(T):
  if t % 100 == 0:
    img = reverse_to_img(x)
    imgs.append(img)

  beta = betas[t]
  eps = torch.randn_like(x)
  x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

plt.figure(figsize=(15, 6))

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'flower.png')
img = plt.imread(file_path)
x = preprocess(img)

T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)

def add_noise(x_0, t, betas):
  T = len(betas)
  assert t >=1 and t <= T

  alphas = 1

