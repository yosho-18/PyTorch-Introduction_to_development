# 学習効率化のTips，Dropoutによる正則化，Batch Normalizationによる学習の加速
# 訓練不足と過学習を抑える
from sklearn.model_selection import train_test_split
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

# 全体の30%は検証用
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

# 層を積み重ねて深いNNを構築する
k = 100
net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),  # 活性化関数
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
# 訓練用データでDataLoaderを作成
ds = TensorDataset(X_train, Y_train)
loader = DataLoader(ds, batch_size=32, shuffle=True)  # 32分割


# TensorDatasetをDataLoaderに渡してデータの一部のみを簡単に受け取れる例

# 最適化を実行
train_losses = []
test_losses = []
for epoch in range(100):
    running_loss = 0.0
    for i, (xx, yy) in enumerate(loader):  # mini-batch学習
        # xx, yyは32個分のみ受け取れる 40
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)
    # テスト部分
    y_pred = net(X_test)
    test_loss = loss_fn(y_pred, Y_test)
    test_losses.append(test_loss.item())


# 正則化（過学習を抑える），いくつかのノード（変数の次元）を意図的に使用しない
# 確率0.5でランダムに変数の次元を捨てるDropoutを各層に追加
net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),  # 活性化関数
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, 10)
)

optimizer = optim.Adam(net.parameters())
# 最適化を実行
train_losses = []
test_losses = []
for epoch in range(100):
    running_loss = 0.0
    # ネットワークを訓練モードにする
    net.train()  # Dropout on
    for i, (xx, yy) in enumerate(loader):  # mini-batch学習
        # xx, yyは32個分のみ受け取れる 40
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)

    # ネットワークを評価モードにして検証データの損失関数を計算する
    net.eval()  # Dropout off
    # テスト部分
    y_pred = net(X_test)
    test_loss = loss_fn(y_pred, Y_test)
    test_losses.append(test_loss.item())


# Linear層にはBatchNorm1dを適用する
net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),  # 活性化関数
    nn.BatchNorm1d(k),  # 多層のとき適度に正規化，学習を安定化，加速させる
    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),
    nn.Linear(k, 10)
)  # Dropoutと同じくtrainとevalで切り替える
