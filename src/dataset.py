import PIL
import numpy
import albumentations


class SiameseVanillaDataset():
    '''
    Author @Pranav Pandey, Date: 04_03_2020.
    This class is for loading dataset from a given folder in pairs with a label given to the pair of images;
    if they are simillar (1) or different (0) to each other.
    '''

    def __init__(self, sigDataset, img_height, img_width, mean, std, transforms=True):
        self.sigDataset = sigDataset

        if transforms:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                scale_limit=0.1,
                                rotate_limit=5,
                                p=0.9),
                albumentations.Normalize(mean, std, always_apply= True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),

                albumentations.Normalize(mean, std, always_apply= True)
            ])
    def __getitem__(self, index):
        pass
    def __len__(self):
        return len(self.sigDataset.imgs)