# 多クラスのロジスティクス回帰
from sklearn.datasets import load_digits
import torch
# %matplotlib inline
from matplotlib import pyplot as plt
from torch import nn, optim


digits = load_digits()

# 0～9までの10種類の手書き文字の数字データセット
X = digits.data
y = digits.target

# NumPyのndarrayをPyTorchのTensorに変換
X = torch.tensor(X, dtype=torch.float32)
# CrossEntropyLoss関数はyとしてint64型のTensorを受け取るので注意
y = torch.tensor(y, dtype=torch.int64)

# 出力は10（クラス数）次元
net = nn.Linear(X.size()[1], 10)

# ソフトマックスエントロピー（多クラスの時はこっち）
loss_fn = nn.CrossEntropyLoss()

# SGD
optimizer = optim.SGD(net.parameters(), lr=0.01)



# 損失関数のログ
losses = []
# 100回のイテレーションを回す
for epoc in range(100):
    # 前回のbackwardメソッドで計算された勾配の値を削除
    optimizer.zero_grad()

    # 線形モデルでyの予測値を計算
    y_pred = net(X)  # wの初期値自動，100個

    # MSE lossを計算
    loss = loss_fn(y_pred, y)  # 適応する損失関数Lが前と違う
    # lossのwによる微分を計算
    loss.backward()
    # 勾配（重み）を更新する
    optimizer.step()

    # 収束確認のためにlossを記録しておく
    losses.append(loss.item())

plt.plot(losses)
print(list(net.parameters()))

# torch.maxは集計軸を指定すると最大値のほかにその位置も返す
_, y_pred = torch.max(net(X), 1)  # ソフトマックス関数は単調増加なのでそのままでOK

# 正解率を計算する
print((y_pred == y).sum().item() / len(y))