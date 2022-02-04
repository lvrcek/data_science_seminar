import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PileogramDataset(Dataset):

    def __init__(self, dir_nonchimeric, dir_chimeric, transform=None):
        self.path_list = []
        self.label_list = []
        self.transform = transform
        self.dir_nonchimeric = dir_nonchimeric
        self.dir_chimeric = dir_chimeric

        for file in os.listdir(dir_nonchimeric):
            self.path_list.append(os.path.join(dir_nonchimeric, file))
            self.label_list.append(0)
        for file in os.listdir(dir_chimeric):
            self.path_list.append(os.path.join(dir_chimeric, file))
            self.label_list.append(1)

    def __len__(self):
        return len(self.path_list)
        # return len(self.path_list)

    def __getitem__(self, idx):
        if type(idx) == int:
            image = Image.open(self.path_list[idx])
            label = self.label_list[idx]
            path = str(self.path_list[idx])
            if self.transform:
                image = self.transform(image)
            sample = {'image': image, 'label': label, 'path': path}
            return sample
        if type(idx) == slice:
            new_ds = PileogramDataset(self.dir_nonchimeric, self.dir_chimeric, self.transform)
            new_ds.path_list = new_ds.path_list[idx]
            new_ds.label_list = new_ds.label_list[idx]
            return new_ds
