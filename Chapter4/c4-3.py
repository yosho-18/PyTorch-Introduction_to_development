# 転移学習：事前に作ったモデルを再利用する
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm

# CNN: VGG, Inception, ResNet
# CIFAR-10, ImageNet

# ImageFolder関数を使用してDatasetを作成する
train_imgs = ImageFolder("./taco_and_burrito/train/",
    transform=transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
)

test_imgs = ImageFolder("./taco_and_burrito/test/",
    transform=transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
)

# DataLoaderを作成
train_loader = DataLoader(train_imgs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_imgs, batch_size=32, shuffle=False)

print(train_imgs.classes)
print(train_imgs.class_to_idx)


# 事前学習済みのresnet18をロード
net = models.resnet18(pretrained=True)

# すべてのパラメータを微分対象外になる
for p in net.parameters():
    p.requires_grad=False

# 最後の線形層を付け加える
fc_input_dim = net.fc.in_features
net.fc = nn.Linear(fc_input_dim, 2)


# c4-1のeval_netと全く同じ
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
    # 予測精度を計算
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

# c4-1のtrain_netとほぼ同じ，最後のところだけ学習させる
def train_net(net, train_loader, test_loader, only_fc=True, optimizer_cls=optim.Adam,
              loss_fn=nn.CrossEntropyLoss(), n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    if only_fc:
        # 最後の線形パラメータのみを，optimizerに渡す
        optimizer = optimizer_cls(net.fc.parameters())
    else:
        optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        # ネットワークを訓練モードにする
        net.train()
        n = 0
        n_acc = 0
        # 非常に時間がかかるのでtqdmを使用してプログレスバーを出す
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss / i)
        # 訓練データの予測精度
        train_acc.append(n_acc / n)
        # 検証データの予測精度
        val_acc.append(eval_net(net, test_loader, device))
        # このepochでの結果を表示
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)


# ネットワークの全パラメータをGPUに転送
# net.to("cuda:0")

# 訓練を実行
train_net(net, train_loader, test_loader, n_iter=20)  # device="cuda:0"


# 入力をそのまま出力するダミー層を作り，fcを置き換える
class IdentityLayer(nn.Module):
    def forward(self, x):
        return x

net = models.resnet18(pretrained=True)
for p in net.parameters():
    p.requires_grad=False
net.fc = IdentityLayer()



# 著者作成CNNモデル

# (N, C, H, W)形式のTensorを(N, C * H * W)に引き伸ばす層
# 畳み込み層の出力をMLPに渡す際に必要
class FlattenLayer(nn.Module):
    def forward(self, x):
        sizes = x.size()
        return x.view(sizes[0], -1)


conv_net = nn.Sequential(
    nn.Conv2d(3, 32, 5),  # 画像の畳み込みを行う
    nn.MaxPool2d(2),  # プーリング（畳み込みの後に位置の感度を鈍くする）を行う
    nn.ReLU(),
    nn.BatchNorm2d(32),

    nn.Conv2d(32, 64, 5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.BatchNorm2d(64),

    nn.Conv2d(64, 128, 5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.BatchNorm2d(128),

    FlattenLayer()
)


# 畳み込みによって最終的にどのようなサイズになっているかを，実際にデータを入れて確認する
test_input = torch.ones(1, 3, 224, 224)  # 28×28
conv_output_size = conv_net(test_input).size()[-1]

# 最終的なCNN
net = nn.Sequential(
    conv_net,
    nn.Linear(conv_output_size, 2)
)


# 訓練を実行
train_net(net, train_loader, test_loader, n_iter=10, only_fc=False)  # device="cuda:0"