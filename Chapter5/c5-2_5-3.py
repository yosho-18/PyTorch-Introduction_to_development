import glob
import pathlib
import re
import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm
from statistics import mean

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression

# 全部で10000種類のトークンを20次元のベクトルで表現する場合
emb = nn.Embedding(10000, 20, padding_idx=0)
# Embedding層への入力はint64のTensor
inp = torch.tensor([1, 2, 5, 2, 10], dtype=torch.int64)
# 出力はfloat32のTensor
out = emb(inp)  # 5×20

# HTMLタグ?
remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;]|<.*?>")  # \:エスケープシーケンス，*?直前のパターンを0回以上繰り返し．最短マッチで、0回以上の繰り返し
shift_marks_regex = re.compile("([?!])")  # ?と!の集合

# 長い文字列をトークンIDのリストに変換
def text2ids(text, vocab_dict):
    # !?以外の記号の削除
    text = remove_marks_regex.sub("", text)
    # !?と単語の間にスペースを挿入
    text = shift_marks_regex.sub(r" \1", text)  # それぞれ検索文字列の1番目の ( ) に一致した文字列*
    tokens = text.split()
    return [vocab_dict.get(token, 0) for token in tokens]  # tokenは単語，IDを返す，存在しなければ0

# IDのリストをint64のTensorに変換
def list2tensor(token_idxes, max_len=100, padding=True):
    if len(token_idxes) > max_len:  # 各文章の分割後のトークンの数を制限
        token_idxes = token_idxes[:max_len]
    n_tokens = len(token_idxes)
    if padding:  # 0埋め
        token_idxes = token_idxes + [0] * (max_len - len(token_idxes))
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens


# Datasetクラスの作成
class IMDBDataset(Dataset):
    def __init__(self, dir_path, train=True, max_len=100, padding=True):
        self.max_len = max_len
        self.padding = padding
        path = pathlib.Path(dir_path)
        vocab_path = path.joinpath("imdb.vocab")

        # ボキャブラリファイルを読み込み，行ごとに分割
        self.vocab_array = vocab_path.open(encoding="utf-8").read().strip().splitlines()  # utf-8_sig
        # 単語をキーとし，値がIDのdictを作る
        self.vocab_dict = dict((w, i + 1) for (i, w) in enumerate(self.vocab_array))

        if train:
            target_path = path.joinpath("train")
        else:
            target_path = path.joinpath("test")
        pos_files = sorted(glob.glob(str(target_path.joinpath("pos/*.txt"))))  # 下のtxt全て取る
        neg_files = sorted(glob.glob(str(target_path.joinpath("neg/*.txt"))))
        # posは1，negは0のlabelを付けて，(file_path, label)のtupleのリストを作成，(label, file_path)
        self.labeled_files = list(zip([0] * len(neg_files), neg_files)) + list(zip([1] * len(pos_files), pos_files))

    @property  # getter：インスタンス変数の値を返す
    def vocab_size(self):
        return len(self.vocab_array)

    def __len__(self):
        return len(self.labeled_files)

    def __getitem__(self, idx):  # 1つずつ取り出すときに呼び出す
        label, f = self.labeled_files[idx]
        # ファイルのテキストデータを読み取って小文字に変換
        data = open(f, encoding="utf-8").read().lower()  # encoding="utf-8"が必要
        # テキストデータをIDのリストに変換
        data = text2ids(data, self.vocab_dict)  # 100個
        # IDのリストをTensorに変換
        data, n_tokens = list2tensor(data, self.max_len, self.padding)
        return data, label, n_tokens

train_data = IMDBDataset("./aclImdb/")
test_data = IMDBDataset("./aclImdb/", train=False)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # , shuffle=True, num_workers=4
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)  # , num_workers=4

class SequenceTaggingNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, hidden_size=50, num_layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None, l=None):
        # IDをEmbeddingで多次元のベクトルに変換する
        # xは(batch_size, step_size) -> (batch_size, step_size, embedding_dim)
        x = self.emb(x)
        # 初期状態h0と共にRNNにxを渡す
        # xは(batch_size, step_size, embedding_dim) -> (batch_size, step_size, hidden_dim)
        x, h = self.lstm(x, h0)
        # 最後のステップのみ取り出す
        # xは(batch_size, step_size, hidden_dim) -> (batch_size, 1)
        if l is not None:
            # 入力のもともとの長さがある場合はそれを使用する
            x = x[list(range(len(x))), l - 1, :]
        else:
            # なければ単純に最後を使用する
            x = x[:, -1, :]
        # 取り出した最後のステップを線形層に入れる
        x = self.linear(x)
        # 余分な次元を削除する
        # (batch_size, 1) -> (batch_size, )
        x = x.squeeze()
        return x





# 評価のヘルパー関数
def eval_net(net, data_loader, device="cpu"):
    # DropoutやBatchNormを無効化
    net.eval()
    ys = []
    ypreds = []
    for x, y, l in data_loader:
        # 長さの配列を長い順にソート（PackedSequenceを使う場合）
        l, sort_idx = torch.sort(l, descending=True)
        # 得られたインデクスを使用してx, yも並べ替え
        x = x[sort_idx]
        y = y[sort_idx]

        # toメソッドで計算を実行するデバイスに転送する
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)
        # 確率が最大のクラスを予測（リスト2.14参照）
        # ここではforward（推論）の計算だけなので，自動微分に必要な処理はoffにして余計な計算を省く
        with torch.no_grad():
            y_pred = net(x, l=l)
            y_pred = (y_pred > 0).long()
            ys.append(y)
            ypreds.append(y_pred)
            """_, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)"""
    # ミニバッチごとの予測結果などを1つにまとめる
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    # 予測精度を計算
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

"""# 訓練のヘルパー関数
def train_net(net, train_loader, test_loader, optimizer_cls=optim.Adam,
              loss_fn=nn.CrossEntropyLoss(), n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
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
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)"""

# num_embeddingには0を含めてtrain_data.vocab_size+1を入れる
net = SequenceTaggingNet(train_data.vocab_size+1, num_layers=2)
# net.to("cuda:0")
opt = optim.Adam(net.parameters())
loss_f = nn.BCEWithLogitsLoss()

for epoch in range(10):
    losses = []
    net.train()
    for x, y, l in tqdm.tqdm(train_loader):
        # x = x.to("cuda:0")
        # y = y.to("cuda:0")
        # l = l.to("cuda:0")
        y_pred = net(x, l=l)
        loss = loss_f(y_pred, y.float())
        net.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_acc = eval_net(net, train_loader)  # , "cuda:0"
    val_acc = eval_net(net, test_loader)  # , "cuda:0"
    print(epoch, mean(losses), train_acc, val_acc)


# RNNを使用しないモデルの生成
train_X, train_y = load_svmlight_file("./aclImdb/train/labeledBow.feat")
test_X, test_y = load_svmlight_file("./aclImdb/test/labeledBow.feat", n_features=train_X.shape[1])

model = LogisticRegression(C=0.1, max_iter=1000)
model.fit(train_X, train_y)
model.score(train_X, train_y), model.score(test_X, test_y)




# 可変長の系列の扱い
# PackedSequenceの性質を利用したモデルの作成
class SequenceTaggingNet2(SequenceTaggingNet):
    def forward(self, x, h0=None, l=None):
        # IDをEmbeddingで多次元のベクトルに変換する
        # xは(batch_size, step_size) -> (batch_size, step_size, embedding_dim)
        x = self.emb(x)

        # 長さ情報が与えられている場合はPackedSequenceを作る
        if l is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)


        # RNNに通す
        x, h = self.lstm(x, h0)

        # 最後のステップのみ取り出して線形層に入れる
        if l is not None:
            # 長さ情報がある場合は最後の層の内部状態のベクトルを直接利用できる
            # LSTMは通常の内部状態の他にブロックセルの状態もあるので内部状態のみを使用する
            hidden_state, cell_state = h
            x = hidden_state[-1]
        else:
            x = x[:, -1, :]
        # 線形層に入れる
        x = self.linear(x).squeeze()

        return x


for epoch in range(10):
    losses = []
    net.train()
    for x, y, l in tqdm.tqdm(train_loader):
        # 長さの配列を長い順にソート
        l, sort_idx = torch.sort(l, descending=True)
        # 得られたインデクスを使用してx, yも並べ替え
        x = x[sort_idx]
        y = y[sort_idx]
        # x = x.to("cuda:0")
        # y = y.to("cuda:0")
        # l = l.to("cuda:0")
        y_pred = net(x, l=l)
        loss = loss_f(y_pred, y.float())
        net.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_acc = eval_net(net, train_loader)  # , "cuda:0"
    val_acc = eval_net(net, test_loader)  # , "cuda:0"
    print(epoch, mean(losses), train_acc, val_acc)