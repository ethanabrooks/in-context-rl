import torch

import data.full_history


def expand_as(x: torch.Tensor, y: torch.Tensor):
    while x.dim() < y.dim():
        x = x[..., None]
    return x.expand_as(y)


class Data(data.full_history.Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unpadded_mask = self.unpadded_mask.cuda()
        self.unpadded_data = self.unpadded_data.cuda()

    def __getitem__(self, idx):
        return self.unpadded_data[idx].flatten(), self.unpadded_mask[idx].flatten()

    def __len__(self):
        return len(self.unpadded_data)
