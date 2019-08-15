# DatasetとDataLoader
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader
import torch
# %matplotlib inline
from matplotlib import pyplot as plt
from torch import nn, optim


digits = load_digits()

# 0～9までの10種類の手書き文字の数字データセット
X = digits.data
Y = digits.target

# NumPyのndarrayをPyTorchのTensorに変換
X = torch.tensor(X, dtype=torch.float32)
# CrossEntropyLoss関数はYとしてint64型のTensorを受け取るので注意
Y = torch.tensor(Y, dtype=torch.int64)

# Datasetを作成
ds = TensorDataset(X, Y)  # X, Yをまとめる

# 異なる順番で64個ずつデータを返すDataLoaderを作成
loader = DataLoader(ds, batch_size=64, shuffle=True)

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),  # 活性化関数
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# TensorDatasetをDataLoaderに渡してデータの一部のみを簡単に受け取れる例

# 最適化を実行
losses = []
for epoch in range(10):
    running_loss = 0.0
    for xx, yy in loader:  # mini-batch学習
        # xx, yyは64個分のみ受け取れる 29
        u = yy.size()
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)