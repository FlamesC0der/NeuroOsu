import base64
import torch
import numpy as np
from torch.utils.data import Dataset

from AI.settings import *
from AI.util.loader import load_data


class OsuDataset(Dataset):
    def __init__(self, group):
        super().__init__()

        if group == "train":
            self.data = load_data(TRAIN_DIR)
        elif group == "test":
            self.data = load_data(TEST_DIR)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = self.data[idx]

        t_img = torch.from_numpy(np.frombuffer(base64.b64decode(dat['img']), dtype=np.float32).copy())
        t_pos = torch.tensor(dat['pos'])

        return {'img': t_img, 'label': t_pos}
