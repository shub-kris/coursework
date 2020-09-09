import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torchvision


class NoisyFashionMNIST(Dataset):
    def __init__(self, incoming_df):

        super(NoisyFashionMNIST, self).__init__()
        self.incoming_df = incoming_df

    def __len__(self):
        return len(self.incoming_df)

    def __getitem__(self, idx):

        # TODO: add noise

        image, target = self.incoming_df[idx]

        noise = torch.rand_like(image)

        noisy_image = noise + image

        # return a pair of data [original_image, noisy_image]
        # see Tutorial for hints
        return image, noisy_image


if __name__ == "__main__":

    fmnist_train = dset.FashionMNIST("./", train=True, transform=transforms.ToTensor())
    fmnist_noisy = NoisyFashionMNIST(fmnist_train)

    for i in range(80):
        # TODO: extract images
        index = random.randint(100, 200)
        dp = fmnist_noisy[index]
        images_vis = torch.cat(
            [dp[1].cpu().repeat(1, 3, 1, 1), dp[0].cpu().repeat(1, 3, 1, 1)], axis=3
        )
        img_grid = torchvision.utils.make_grid(images_vis.cpu())
    # TODO: visualize (noisy_image, clean_image) in a grid
