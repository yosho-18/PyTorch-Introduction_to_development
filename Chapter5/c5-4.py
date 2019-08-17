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

import string

# すべてのascii文字で辞書を作る
all_chars = string.printable
vocab_size =len(all_chars)
vocab_dict = dict((c, i) for (i, c) in enumerate(all_chars))

# 文字列を数値のリストに変換する関数
def str2ints(s, vocab_dict):
    return [vocab_dict[c] for c in s]

# 数値のリストを文字列に変換する関数
def ints2str(x, vocab_array):
    return "".join([vocab_array[i] for i in x])

class ShakespeareDataset(Dataset):
    def __init__(self, path, chunk_size=200):
        # ファイルを読み込み，数値のリストに変換する
        data = str2ints(open(path).read().strip(), vocab_dict)
        # Tensorに変換し，splitする
        data = torch.tensor(data, dtype=torch.int64).split(chunk_size)

        # 最後のchunkの長さをチェックして足りない場合には捨てる
        if len(data[-1]) < chunk_size:
            data = data[:-1]

        self.data = data
        self.n_chunks = len(self.data)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        return self.data[idx]

ds = ShakespeareDataset("./tinyshakespeare.txt", chunk_size=200)
loader = DataLoader(ds, batch_size=32, shuffle=True)  # , num_workers=4

class SequenceGenerationNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, hidden_size=50, num_layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # 語xから語yを予測するので，
        # Linearのoutputのサイズは最初のEmbeddingのinputサイズと同じnum_embeddings
        self.linear = nn.Linear(hidden_size, num_embeddings)

    def forward(self, x, h0=None, l=None):
        x = self.emb(x)
        x, h = self.lstm(x, h0)
        x = self.linear(x)
        return x, h

    def generate_seq(net, start_phrase="The King said", length=200, temperature=0.8, device="cpu"):
        # モデルを評価モードにする
        net.eval()
        # 出力の数値を格納するリスト
        result = []

        # 開始文字列をTensorに変換
        start_tensor = torch.tensor(str2ints(start_phrase, vocab_dict), dtype=64).to(device)
        # 先頭にbatch次元を付ける
        x0 = start_tensor.unsqueeze(0)
        # RNNに通して出力と新しい内部状態を得る
        o, h = net(x0)
        # 出力を（正規化されていない）確率に変換
        out_dist = o[:, -1].view(-1).exp()
        # 確率から実際のインデクスをサンプリング
        top_i = torch.multinomial(out_dist, 1)[0]  # 出力の指数を取った値で定義される多項分布からサンプリングを行うことで次に来る単語を予測
        # 結果を保存
        result.append(top_i)

        # 生成された結果を次々にRNNに入力していく
        for i in range(length):
            inp = torch.tensor([[top_i]], dtype=torch.int64)
            inp = inp.to(device)
            