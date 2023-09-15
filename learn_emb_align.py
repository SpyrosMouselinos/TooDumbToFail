import torch
import torch.nn as nn
import numpy as np
import json
from pstp_net.feat_script.clip import load as clip_load
from pstp_net.feat_script.clip import tokenize as clip_tokenize
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

#device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip_load("ViT-B/32", device=device)


class AlignEmb(nn.Module):
    def __init__(self):
        super(AlignEmb, self).__init__()
        self.loss_fn = torch.nn.functional.l1_loss
        self.linear_proj = nn.Linear(512, 512)

    def calc_loss(self, clip, learned):
        return self.loss_fn(clip, learned)

    def forward(self, clip, learned=None):
        projected = self.linear_proj(clip)
        if learned is not None:
            loss = self.calc_loss(projected, learned)
            return projected, loss
        return projected, None


def qst_feat_extract(qst):
    text = clip_tokenize(qst).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features[0]


emb_matrix = np.load('emb_512_overlap.npy')
translation_json = json.load(open('./data/answer_keysword_overlap.json', 'r'))

amodel = AlignEmb()
amodel.to(device)
optim = AdamW(params=amodel.parameters(), lr=0.0001)

optim.zero_grad()
curr_loss = 0
clip_mat = []
learned_mat = []
for entry in translation_json:
    clip_mat.append(qst_feat_extract(entry).float())
    learned_mat.append(torch.Tensor(emb_matrix[translation_json[entry]['int_index'], :][None, ...]).float())


class CustomDs(Dataset):
    def __init__(self, c, l):
        self.c = c
        self.l = l
        return

    def __len__(self):
        return len(self.c)

    def __getitem__(self, idx):
        return self.c[idx], self.l[idx]

t = int(len(clip_mat) * 0.8)
train_dataset = CustomDs(clip_mat[:t], learned_mat[:t])
val_dataset = CustomDs(clip_mat[t+1:], learned_mat[t+1:])
train_dl = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

for e in range(20):
    print(f"Epoch: {e}")
    curr_loss = 0
    step = 0
    for c, l in train_dl:
        word_clip = c.to(device)
        word_emb_idx = l.to(device)
        sugg, loss = amodel(word_clip, word_emb_idx)
        loss.backward()
        optim.step()
        optim.zero_grad()
        curr_loss += loss.detach().cpu().item()
        step += 1
        print(f"MSE Loss: {curr_loss / step}")
    curr_loss = 0
    step = 0
    with torch.no_grad():
        for c, l in val_dl:
            word_clip = c.to(device)
            word_emb_idx = l.to(device)
            sugg, loss = amodel(word_clip, word_emb_idx)
            curr_loss += loss.detach().cpu().item()
            step += 1
            print(f"Val MSE Loss: {curr_loss / step}")

torch.save(amodel.state_dict(), 'proj_512.pth')