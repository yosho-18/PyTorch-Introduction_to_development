import numpy as np
import torch

# 入れ子のlistを渡して作成
t = torch.tensor([[1, 2], [3, 4]])
print(t)

# deviceを指定することでGPUにTensorを作成する
# t = torch.tensor([[1, 2], [3, 4]], device="cuda:0")
# print(t)

# dtypeを指定することで倍精度のTensorを作る
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
print(t)

# 0から9まで数値で初期化された1次元のTensor
t = torch.arange(0, 10)
print(t)

#
