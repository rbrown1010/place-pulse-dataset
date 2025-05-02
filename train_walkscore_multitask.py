#!/usr/bin/env python
"""
train_walkscore_multitask.py
----------------------------

Train a single VGG-16 CNN that predicts five Place Pulse perception
scores, then save:
    • walkscore_multitask.pth   – model weights
    • walkscore_meta.json       – min / max label stats + blend weights
"""

# ----------------------------------------------------#
# 0.  Imports & settings                              #
# ----------------------------------------------------#
import json, os, random
import numpy as np
import pandas as pd

import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from skimage import io
from tqdm import tqdm

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
BATCH_SIZE = 32
EPOCHS     = 6
LR         = 1e-4
HEADS      = ['safe', 'lively', 'clean', 'beautiful', 'depressing']
IMG_DIR    = 'data/images/'
QSCORES_TSV_PATH = 'data/qscores.tsv'
WEIGHTS_FILE     = 'walkscore_multitask.pth'
META_FILE        = 'walkscore_meta.json'

WALK_WEIGHTS = {         # heuristic linear blend
    'safe'      : 0.30,
    'lively'    : 0.25,
    'clean'     : 0.20,
    'beautiful' : 0.15,
    'depressing': -0.10
}

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed()

# ----------------------------------------------------#
# 1.  Build wide dataframe (one row = one image)      #
# ----------------------------------------------------#
STUDY_IDS = {
    'safe'      : '50a68a51fdc9f05596000002',
    'lively'    : '50f62c41a84ea7c5fdd2e454',
    'clean'     : '50f62c68a84ea7c5fdd2e456',
    'wealthy'   : '50f62cb7a84ea7c5fdd2e458',
    'depressing': '50f62ccfa84ea7c5fdd2e459',
    'beautiful' : '5217c351ad93a7d3e7b07a64'
}

if __name__ == '__main__':
    print('Reading qscores.tsv …')
    qdf = pd.read_csv(QSCORES_TSV_PATH, sep='\t')
    qdf = qdf[qdf.study_id.isin([STUDY_IDS[h] for h in HEADS])]
    wide = qdf.pivot_table(index='location_id',
                           columns='study_id',
                           values='trueskill.score')
    wide.columns = [dict((v, k) for k, v in STUDY_IDS.items())[c]
                    for c in wide.columns]
    wide = wide.dropna()

    # label scaling
    mins, maxs = wide.min(), wide.max()
    wide_norm  = (wide - mins) / (maxs - mins)

    # ----------------------------------------------------#
    # 2.  Train / val split                               #
    # ----------------------------------------------------#
    train_ids, val_ids = train_test_split(wide_norm.index,
                                          test_size=0.2,
                                          random_state=42)

    # ----------------------------------------------------#
    # 3.  Torch dataset                                   #
    # ----------------------------------------------------#
    transform = T.Compose([
        T.ToPILImage(),
        T.Lambda(lambda im: im.crop((0, 0, im.width, im.height-25))),  # strip bar
        T.Resize(256, interpolation=3),
        T.CenterCrop(224),
        T.ToTensor(),                                                  # [0-1]
        T.Normalize(mean=[0.485, 0.456, 0.406],                        # ImageNet
                    std =[0.229, 0.224, 0.225])
    ])

    class WalkDataset(Dataset):
        def __init__(self, loc_ids):
            self.loc_ids = list(loc_ids)
        def __len__(self): return len(self.loc_ids)
        def __getitem__(self, idx):
            loc = self.loc_ids[idx]
            img = io.imread(os.path.join(IMG_DIR, f'{loc}.jpg'))
            img = transform(img)
            y   = torch.tensor(wide_norm.loc[loc, HEADS].values,
                               dtype=torch.float32)
            return img, y

    train_loader = DataLoader(WalkDataset(train_ids), BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(WalkDataset(val_ids),   BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ----------------------------------------------------#
    # 4.  VGG-16 multi-head model                         #
    # ----------------------------------------------------#
    class VGGMulti(nn.Module):
        def __init__(self, heads):
            super().__init__()
            base = models.vgg16(weights='DEFAULT')
            self.features = base.features
            self.avgpool  = base.avgpool
            self.flatten  = nn.Flatten()
            dim = 512 * 7 * 7
            self.heads = nn.ModuleDict({
                h: nn.Sequential(
                    nn.Linear(dim, 256), nn.ReLU(inplace=True),
                    nn.Linear(256, 1)
                ) for h in heads
            })
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            return torch.cat([self.heads[h](x) for h in HEADS], dim=1)

    model     = VGGMulti(HEADS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ----------------------------------------------------#
    # 5.  Training loop                                   #
    # ----------------------------------------------------#
    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train()
        tr_loss = 0.
        for imgs, ys in tqdm(train_loader,
                             desc=f'Epoch {epoch:02d}/{EPOCHS} – train'):
            imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), ys)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(imgs)
        tr_loss /= len(train_loader.dataset)

        # ---- val ----
        model.eval()
        vl_loss = 0.
        with torch.no_grad():
            for imgs, ys in tqdm(val_loader, desc='val'):
                imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
                vl_loss += criterion(model(imgs), ys).item() * len(imgs)
        vl_loss /= len(val_loader.dataset)
        print(f'>> epoch {epoch:02d} | train {tr_loss:.4f} | val {vl_loss:.4f}')

    # ----------------------------------------------------#
    # 6.  Save model and meta                             #
    # ----------------------------------------------------#
    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f'Saved weights → {WEIGHTS_FILE}')

    meta = {
        "heads"   : HEADS,
        "mins"    : {h: float(mins[h]) for h in HEADS},
        "maxs"    : {h: float(maxs[h]) for h in HEADS},
        "weights" : WALK_WEIGHTS
    }
    with open(META_FILE, "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f'Saved meta    → {META_FILE}') 