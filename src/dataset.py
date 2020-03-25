import PIL
import numpy
import albumentations
import random
import torch
from albumentations.pytorch.transforms import ToTensor
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SiameseVanillaDataset():
    '''
    Author @Pranav Pandey, Date: 04_03_2020.
    This class is for loading dataset from a given folder in pairs with a label given to the pair of images;
    if they are simillar (1) or different (0) to each other.
    '''

    def __init__(self, imageFolderDataset, img_height, img_width, mean, std, transform=False):
        self.imageFolderDataset = imageFolderDataset    

        if transform:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                scale_limit=0.1,
                                rotate_limit=5,
                                p=0.9),
                albumentations.Normalize(mean, std, always_apply= True),
                ToTensor()
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply= True),
                ToTensor()
            ])
            self.aug_2 = transforms.Compose([transforms.Resize((520,200)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])
    
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = self.aug_2(img0)
        img1 = self.aug_2(img1)
        # img0 = torch.from_numpy(np.moveaxis(img0 / (255.0 if img0.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        # img1 = torch.from_numpy(np.moveaxis(img1 / (255.0 if img1.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]==img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)