# MLPの構築と学習
from sklearn.datasets import load_digits
import torch
# %matplotlib inline
from matplotlib import pyplot as plt
from torch import nn, optim


# nn.Sequential:層が一直線に積み重なった形のNNをFeedforward型という
net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),  # 活性化関数
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)
"""
# 出力は10（クラス数）次元
net = nn.Linear(X.size()[1], 10)"""


digits = load_digits()

# 0～9までの10種類の手書き文字の数字データセット
X = digits.data
Y = digits.target

# NumPyのndarrayをPyTorchのTensorに変換
X = torch.tensor(X, dtype=torch.float32)
# CrossEntropyLoss関数はYとしてint64型のTensorを受け取るので注意
Y = torch.tensor(Y, dtype=torch.int64)


# ソフトマックスエントロピー（多クラスの時はこっち）
loss_fn = nn.CrossEntropyLoss()

# SGD
optimizer = optim.Adam(net.parameters())



# 損失関数のログ
losses = []
# 500回のイテレーションを回す
for epoc in range(500):
    # 前回のbackwardメソッドで計算された勾配の値を削除
    optimizer.zero_grad()

    # 線形モデルでyの予測値を計算
    y_pred = net(X)  # wの初期値自動，100個

    # MSE lossを計算
    loss = loss_fn(y_pred, Y)  # 適応する損失関数Lが前と違う
    # lossのwによる微分を計算
    loss.backward()
    # 勾配（重み）を更新する
    optimizer.step()

    # 収束確認のためにlossを記録しておく
    losses.append(loss.item())

plt.plot(losses)
#print(list(net.parameters()))


# toメソッドでGPUに転送
"""X = X.to("cuda:0")
Y = Y.to("cuda:0")
net.to("cuda:0")"""

# 以下同様にoptimizerをセットし学習ループを回す