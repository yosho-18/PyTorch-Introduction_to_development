# ネットワークのモジュール化
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader
import torch
# %matplotlib inline
from matplotlib import pyplot as plt
from torch import nn, optim

# 独自のネットワーク層（カスタム層）を作る
# 活性化関数ReLUとDropoutを含んだカスタムの線形層を作り，それを用いてMLPを記述
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.relu = nn.Relu()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

# nn.Sequential
mlp = nn.Sequential(
    CustomLinear(64, 200),
    CustomLinear(200, 200),
    CustomLinear(200, 200),
    nn.Linear(200, 10)
)


# nn.Moduleを継承したクラスの利用（nn.Sequentialを使わない）
class MyMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ln1 = CustomLinear(in_features, 200)
        self.ln2 = CustomLinear(200, 200)
        self.ln3 = CustomLinear(200, 200)
        self.ln4 = CustomLinear(200, out_features)

    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.ln4(x)
        return x

mlp = MyMLP(64, 10)
