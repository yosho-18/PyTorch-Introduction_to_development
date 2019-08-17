import numpy as np
import torch

# 入れ子のlistを渡して作成
t = torch.tensor([[1, 2], [3, 4.]]); print(t)
# deviceを指定することでGPUにTensorを作成する
# t = torch.tensor([[1, 2], [3, 4.]], device="cuda:0")
# print(t)
# dtypeを指定することで倍精度のTensorを作る
t = torch.tensor([[1, 2], [3, 4.]], dtype=torch.float64); print(t)
# 0から9まで数値で初期化された1次元のTensor
t = torch.arange(0, 10); print(t)
# すべての値が0の100×10のTensorを作成し，toメソッドでGPUに転送する
# t = torch.zeros(100, 10).to("cuda:0")
# 正規乱数で100×10のTensorを作成
t = torch.randn(100, 10); print(t)
# Tensorのshapeはsizeメソッドで習得可能
print(t.size(), "サイズ")


# numpyメソッドを使用してndarrayに変換
t = torch.tensor([[1, 2], [3, 4.]])
x = t.numpy()
print(x)
# GPU上のTensorはcpuメソッドで，一度CPUのTensorに変換する必要がある
# t = torch.tensor([[1, 2], [3, 4.]], device="cuda:0")
# x = to("cpu").numpy()


t = torch.tensor([[1, 2, 3], [4, 5, 6.]])

# スカラーの添字で指定
print(t[0, 2], "スカラーの添字で指定")
# スライスで指定
print(t[:, :2])
# 添字のリストで指定
print(t[:, [1, 2]])  # 1行目と2行目の全てを取り出す
# マスク配列を使用して3より大きい部分のみ選択
print(t[t > 3])
# [0, 1]要素を100にする
t[0, 1] = 100;  print(t)
# スライスを使用した一括代入
t[:, 1] = 200; print(t)
# マスク配列を使用して特定条件の要素のみ置換
t[t > 10] = 20; print(t)


# 長さ3のベクトル
v = torch.tensor([1, 2, 3.])
w = torch.tensor([0, 10, 20.])
# 2×3の行列
m = torch.tensor([[0, 1, 2], [100, 200, 300.]])
# ベクトルとスカラーの足し算
v2 = v + 10; print(v2, "ベクトルとスカラーの足し算")
# 累乗も同様
v2 = v ** 2; print(v2)
# 同じ長さのベクトル同士の引き算
z = v - w; print(z)
# 複数の組み合わせ
u = 2 * v - w / 10 + 6.0; print(u)
# 行列とスカラー
m2 = m * 2.0; print(m2)
# 行列とベクトル，(2, 3)の行列と(3, )のベクトルなのでブロードキャストが働く
m3 = m + v; print(m3)
# 行列同士
m4 = m + m; print(m4)


# 100×10のテストデータを使用
X = torch.randn(100, 10)
X = torch.tensor([[10, 20, 30], [40, 50, 60.]])
# 数学関数を含めた数式
y = X * 2 + torch.abs(X); print(y, "数学関数を含めた数式")
# 平均値を求める
m = torch.mean(X); print(m)
# 関数ではなく，メソッドとしても利用できる
m = X.mean(); print(m)
# 集計結果は0次元のTensorでitemメソッドを使用して，値を取り出すことが出来る
m_value = m.item(); print(m_value)  # 平均
# 集計は次元を指定できる．以下の行方向に，集計して列ごとに平均値を計算している
m2 = X.mean(0); print(m2)  # 縦方向に足す


x1 = torch.tensor([[1, 2], [3, 4.]])  # 2×2
x2 = torch.tensor([[10, 20, 30], [40, 50, 60.]])  # 2×3

# 2×2を4×1に見せる
print(x1.view(4, 1), "2×2を4×1に見せる")  # torchのviewはnumpyのreshapeと同じ
# -1は残りの次元を表し，一度だけ使用できる
# 以下の例では-1とすると自動的に4になる
print(x1.view(1, -1))
# 2×3を転置して3×2にする
print(x2.t())
# dim=1に対して結合することで2×5のTensorを作る
# RuntimeError: Expected object of scalar type Long but got scalar type Float for sequence element 1 in sequence argument at position #1 'tensors'
print(torch.cat([x1, x2], dim=1))
# HWCをCHWに変換
# 64×32×3のデータが100個
hwc_img_data = torch.rand(3, 4, 2, 3); print(hwc_img_data)
chw_img_data = hwc_img_data.transpose(1, 2).transpose(1, 3); print(chw_img_data)


m = torch.randn(100, 10)
v = torch.randn(10)

# 内積
d = torch.dot(v, v); print(d, "内積")
# 100×10の行列と長さ10のベクトルとの内積，結果は長さ100のベクトル
v2 = torch.mv(m, v); print(v2)
# 行列積
m2 = torch.mm(m.t(), m); print(m2)
# 特異値分解
u, s, v = torch.svd(m)  # u，vは直交行列，sは対角成分のみの正方行列．最小二乗法を解く，行列の近似，圧縮
print(u, s, v)


x = torch.randn(100, 3)
# 微分の変数として扱う場合はrequires_gradフラグをTrueにする
a = torch.tensor([1, 2, 3.], requires_grad=True); print(a, "requires_gradフラグをTrue")
# 計算をすることで自動的に計算フラグが構築されていく
y = torch.mv(x, a); print(y)  # 100個
o = y.sum(); print(o)
# 微分を実行する
print(o.backward())
# 解析解と比較
print(a.grad != x.sum(0))
print(a.grad, x.sum(0))