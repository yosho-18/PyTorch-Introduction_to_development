# PyTorchで線形回帰モデル（nn, optimモジュールの使用）
import torch
# %matplotlib inline
from matplotlib import pyplot as plt
from torch import nn  # モデルの構築
from torch import optim  # 最適化


# 真の係数
w_true = torch.Tensor([1, 2, 3])
# Xのデータの準備．切片を回帰係数に含めるため，Xの最初の次元に1を追加しておく
X = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)
# 真の係数と各Xとの内積を行列とベクトルの席でまとめて計算
y = torch.mv(X, w_true) + torch.randn(100) * 0.5


# Linear層を作成．今回は切片項は回帰係数に含めるので
# 入力の次元を3とし，bias（切片）をFalseにする
net = nn.Linear(in_features=3, out_features=1, bias=False)  # 線形結合を計算，回帰係数や切片項などのパラメータ

# SGDのオプティマイザーに上で定義したネットワークのパラメータを渡して初期化
optimizer = optim.SGD(net.parameters(), lr=0.1)  # lr:学習率

# MSE lossクラス
loss_fn = nn.MSELoss()  # 平均二乗誤差



# 損失関数のログ
losses = []
# 100回のイテレーションを回す
for epoc in range(100):
    # 前回のbackwardメソッドで計算された勾配の値を削除
    optimizer.zero_grad()

    # 線形モデルでyの予測値を計算
    y_pred = net(X)  # wの初期値自動，100個

    # MSE lossを計算
    # y_predは(n, 1)のようなshapeを持っているので(n, )に直す必要がある（二次元配列を一次元配列にする）
    y_pred = y_pred.view_as(y)
    loss = loss_fn(y_pred, y)

    # lossのwによる微分を計算
    loss.backward()

    # 勾配を更新する
    optimizer.step()

    # 収束確認のためにlossを記録しておく
    losses.append(loss.item())

plt.plot(losses)
print(list(net.parameters()))