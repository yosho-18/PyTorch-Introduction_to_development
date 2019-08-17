# CNN回帰モデルによる画像の高解像度化
import math
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torchvision.utils import save_image
import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm

class DownSizedPairImageFolder(ImageFolder):
    def __init__(self, root, transform=None, large_size=128, small_size=32, **kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Resize(large_size)
        self.small_resizer = transforms.Resize(small_size)

    def __getitem(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        #読み取った画像を128×128ピクセルと32×32ピクセルにリサイズする
        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)

        # その他の変換を適用する
        if self.transform is not None:
            large_img = self.transform(large_img)
            small_img = self.transform(small_img)

        # 32ピクセルの画像と128ピクセルの画像を返す
        return small_img. large_img

train_data = DownSizedPairImageFolder("./lfw_deepfunneled/train", transform=transforms.ToTensor())
test_data = DownSizedPairImageFolder("./lfw_deepfunneled/test", transform=transforms.ToTensor())

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # , num_workers=4
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)  # , num_workers=4


net = nn.Sequential(
    nn.Conv2d(3, 256, 4, stride=2, padding=1),  # 画像の畳み込みを行う
    nn.ReLU(),
    nn.BatchNorm2d(256),

    nn.Conv2d(256, 512, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(512),

    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),

    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(128),

    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),

    nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
)


# PSNR（損失関数）の計算
def psnr(mse, max_v=1.0):
    return 10 * math.log10(max_v ** 2 / mse)

# c4-1のeval_netとほぼ同じ，最後のscoreのみ違う
def eval_net(net, data_loader, device="cpu"):
    # DropoutやBatchNormを無効化
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        # toメソッドで計算を実行するデバイスに転送する
        x = x.to(device)
        y = y.to(device)
        # 確率が最大のクラスを予測（リスト2.14参照）
        # ここではforward（推論）の計算だけなので，自動微分に必要な処理はoffにして余計な計算を省く
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
    # ミニバッチごとの予測結果などを1つにまとめる
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    # 予測精度（MSE）を計算
    score = nn.functional.mse_loss(ypreds, ys).item()
    return score

# c4-1のtrain_netとほぼ同じ，n_accがscoreになっている
def train_net(net, train_loader, test_loader, optimizer_cls=optim.Adam,
              loss_fn=nn.MSELoss(), n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        # ネットワークを訓練モードにする
        net.train()
        n = 0
        score = 0
        # 非常に時間がかかるのでtqdmを使用してプログレスバーを出す
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)

            """_, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item()"""

        train_losses.append(running_loss / len(train_loader))  # i = len(train_loader) ??
        # 検証データの予測精度
        val_acc.append(eval_net(net, test_loader, device))
        # このepochでの結果を表示
        print(epoch, train_losses[-1], psnr(train_losses[-1]), psnr(val_acc[-1]), flush=True)


# ネットワークの全パラメータをGPUに転送
# net.to("cuda:0")

# 訓練を実行
train_net(net, train_loader, test_loader)  # device="cuda:0"



# テストのデータセットからランダムに4つずつ取り出すDataLoader
random_test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
# DataLoaderを1pythonのイテレータに変換し，4つ例を取り出す
it = iter(random_test_loader)
x, y = next(it)

# Bilinearで拡大
bl_recon = torch.nn.functional.upsample(x, 128, mode="bilinear", align_coners=True)
# CNNで拡大
# yp = net(x.to("cuda:0")).to("cpu")
yp = net(x.to("cpu")).to("cpu")

# torch.catでオリジナル，Bilinear，CNNを結合し，save_imageで画像ファイルに書き出し
save_image(torch.cat([y, bl_recon, yp], 0), "cnn_upscale.jpg", nrow=4)