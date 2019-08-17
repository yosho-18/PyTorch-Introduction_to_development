# PyTorchでロジスティクス回帰
from sklearn.datasets import load_iris
import torch
# %matplotlib inline
from matplotlib import pyplot as plt
from torch import nn, optim


iris = load_iris()

# irisは(0, 1, 2)の3クラスの分類問題なのでここは
# (0, 1)の2クラス分のデータだけを使用する
# 本来は訓練用とテスト用に分けるべきだがここでは省略
X = iris.data[:100]
y = iris.target[:100]

# NumPyのndarrayをPyTorchのTensorに変換
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# irisのデータは4次元
net = nn.Linear(4, 1)

# シグモイド関数を作用させ，2クラス分類のクロスエントロピーを計算する関数
loss_fn = nn.BCEWithLogitsLoss()

# SGD（少し大きめの学習率）
optimizer = optim.SGD(net.parameters(), lr=0.25)



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
    loss = loss_fn(y_pred, y)  # 適応する損失関数Lが前と違う

    # lossのwによる微分を計算
    loss.backward()
    # 勾配（重み）を更新する
    optimizer.step()

    # 収束確認のためにlossを記録しておく
    losses.append(loss.item())

plt.plot(losses)
print(list(net.parameters()))


# 線形結合の結果
h = net(X)

# シグモイド関数を作用させた結果はy = 1の確率を返す
prob = nn.functional.sigmoid(h)
# 確率が0.5以上のものをクラス1と予想し，それ以外を0とする
# PyTorchにはBool型がないので対応する型としてByteTensorが出力される
y_pred = prob > 0.5  # Trueなら1，Falseなら0を返す

# 予測結果の確認（yはFloatTensorなのでByteTensorに変換してから比較する）
print((y.byte() == y_pred.view_as(y)).sum().item())