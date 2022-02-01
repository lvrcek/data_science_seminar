import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PileogramDataset(Dataset):

    def __init__(self, dir_nonchimeric, dir_chimeric, transform=None):
        self.path_list = []
        self.label_list = []
        self.transform = transform

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
        if idx < len(self.path_list):
            image = Image.open(self.path_list[idx])
            label = self.label_list[idx]
            path = str(self.path_list[idx])
            if self.transform:
                image = self.transform(image)
            sample = {'image': image, 'label': label, 'path': path}
            return sample
        else:
            pass
            # horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
            # image = Image.open(self.path_list[idx - len(self.path_list)])
            # image = horizontal_flip(image)
            # label = self.label_list[idx - len(self.path_list)]
            # path = str(self.path_list[idx - len(self.path_list)])
            # path = path[:-4] + '_flipped' + path[-4:]
            # if self.transform:
            #     image = self.transform(image)
            # sample = {'image': image, 'label': label, 'path': path}
            # return sample