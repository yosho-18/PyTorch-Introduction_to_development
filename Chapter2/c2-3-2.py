# PyTorchで線形回帰モデル（from scratch）
import torch
# %matplotlib inline
from matplotlib import pyplot as plt

# 真の係数
w_true = torch.Tensor([1, 2, 3])
# Xのデータの準備．切片を回帰係数に含めるため，Xの最初の次元に1を追加しておく
X = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)
# 真の係数と各Xとの内積を行列とベクトルの席でまとめて計算
y = torch.mv(X, w_true) + torch.randn(100) * 0.5
# 勾配降下で最適化するためのパラメータのTensorを乱数で初期化
w = torch.randn(3, requires_grad=True)
# 学習率γ
gamma = 0.1


# 損失関数のログ
losses = []
# 100回のイテレーションを回す
for epoc in range(100):
    # 前回のbackwardメソッドで計算された勾配の値を削除
    w.grad = None

    # 線形モデルでyの予測値を計算
    y_pred = torch.mv(X, w)  # 100個

    # MSE lossとwによる微分を計算
    loss = torch.mean((y - y_pred) ** 2)
    loss.backward()

    # 勾配を更新する
    # wをそのまま代入し更新すると異なるTensorになって，計算グラフが破壊されてしまうのでdataだけを更新する
    w.data = w.data - gamma * w.grad.data
    # w = w - gamma * w.grad

    # 収束確認のためにlossを記録しておく
    losses.append(loss.item())

plt.plot(losses)
print(w)