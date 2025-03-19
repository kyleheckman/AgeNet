import torch
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
import os
from PIL import Image

class UTKFace(Dataset):
    def __init__(
            self,
            path: str
    ):
        self.path = path
        self.ids = os.listdir(path)
    
    def __getitem__(self, index):
        fn = f'{self.path}/{self.ids[index]}'
        img = pil_to_tensor(Image.open(fn)).to(torch.float)
        # Normalize pixel values to [0,1]
        img = img/255

        age = int(self.ids[index].split('_')[0])
        age_onehot = torch.tensor([1 if i == age else 0 for i in range(8,91)]).to(torch.float)

        return age_onehot, img

    def __len__(self):
        return len(self.ids)