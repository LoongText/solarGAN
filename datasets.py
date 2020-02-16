import glob
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):

        file = self.files[index % len(self.files)]
        img = Image.open(file)
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_A = img_A.convert('L')
        img_B = img.crop((w/2, 0, w, h))
        img_B = img_B.convert('L')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        img_C = img_A
        img_A = img_B
        img_B = img_C

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)


class TestDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='test'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'test':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_A = img_A.convert('L')
        img_B = img.crop((w/2, 0, w, h))
        img_B = img_B.convert('L')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
