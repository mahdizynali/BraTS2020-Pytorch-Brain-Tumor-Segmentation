import os
import time
from random import randint

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from IPython.display import Image as show_gif

import seaborn as sns
import imageio
from skimage.transform import resize
from skimage.util import montage
from skimage.transform import rotate
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

#from volumentations import *
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from albumentations import HorizontalFlip, VerticalFlip, Normalize, Compose


import warnings
warnings.simplefilter("ignore")

# class Image3dToGIF3d:
    
#     def __init__(self, 
#                  img_dim: tuple = (55, 55, 55),
#                  figsize: tuple = (15, 10),
#                  binary: bool = False,
#                  normalizing: bool = True,
#                 ):
#         self.img_dim = img_dim
#         print(img_dim)
#         self.figsize = figsize
#         self.binary = binary
#         self.normalizing = normalizing

#     def _explode(self, data: np.ndarray):
       
#         shape_arr = np.array(data.shape)
#         size = shape_arr[:3] * 2 - 1
#         exploded = np.zeros(np.concatenate([size, shape_arr[3:]]),
#                             dtype=data.dtype)
#         exploded[::2, ::2, ::2] = data
#         return exploded
    
#     def _expand_coordinates(self, indices: np.ndarray):
#         x, y, z = indices
#         x[1::2, :, :] += 1
#         y[:, 1::2, :] += 1
#         z[:, :, 1::2] += 1
#         return x, y, z
    
#     def _normalize(self, arr: np.ndarray):
#         arr_min = np.min(arr)
#         return (arr - arr_min) / (np.max(arr) - arr_min)

    
#     def _scale_by(self, arr: np.ndarray, factor: int):
        
#         mean = np.mean(arr)
#         return (arr - mean) * factor + mean
    
#     def get_transformed_data(self, data: np.ndarray):
#         if self.binary:
#             resized_data = resize(data, self.img_dim, preserve_range=True)
#             return np.clip(resized_data.astype(np.uint8), 0, 1).astype(np.float32)
            
#         norm_data = np.clip(self._normalize(data)-0.1, 0, 1) ** 0.4
#         scaled_data = np.clip(self._scale_by(norm_data, 2) - 0.1, 0, 1)
#         resized_data = resize(scaled_data, self.img_dim, preserve_range=True)
        
#         return resized_data
    
#     def plot_cube(self,
#                   cube,
#                   title: str = '', 
#                   init_angle: int = 0,
#                   make_gif: bool = False,
#                   path_to_save: str = 'filename.gif'
#                  ):
       
#         if self.binary:
#             facecolors = cm.winter(cube)
#             print("binary")
#         else:
#             if self.normalizing:
#                 cube = self._normalize(cube)
#             facecolors = cm.gist_stern(cube)
#             print("not binary")
            
#         facecolors[:,:,:,-1] = cube
#         facecolors = self._explode(facecolors)

#         filled = facecolors[:,:,:,-1] != 0
#         x, y, z = self._expand_coordinates(np.indices(np.array(filled.shape) + 1))

#         with plt.style.context("dark_background"):

#             fig = plt.figure(figsize=self.figsize)
#             ax = fig.gca(projection='3d')

#             ax.view_init(30, init_angle)
#             ax.set_xlim(right = self.img_dim[0] * 2)
#             ax.set_ylim(top = self.img_dim[1] * 2)
#             ax.set_zlim(top = self.img_dim[2] * 2)
#             ax.set_title(title, fontsize=18, y=1.05)

#             ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

#             if make_gif:
#                 images = []
#                 for angle in tqdm(range(0, 360, 5)):
#                     ax.view_init(30, angle)
#                     fname = str(angle) + '.png'

#                     plt.savefig(fname, dpi=120, format='png', bbox_inches='tight')
#                     images.append(imageio.imread(fname))
#                     #os.remove(fname)
#                 imageio.mimsave(path_to_save, images)
#                 plt.close()
#             else:
#                 plt.show()
                
# def merging_two_gif(path1: str, path2: str, name_to_save: str):

#     gif1 = imageio.get_reader(path1)
#     gif2 = imageio.get_reader(path2)

#     #If they don't have the same number of frame take the shorter
#     number_of_frames = min(gif1.get_length(), gif2.get_length()) 

#     #Create writer object
#     new_gif = imageio.get_writer(name_to_save)

#     for frame_number in range(number_of_frames):
#         img1 = gif1.get_next_data()
#         img2 = gif2.get_next_data()
#         new_image = np.hstack((img1, img2))
#         new_gif.append_data(new_image)

#     gif1.close()
#     gif2.close()    
#     new_gif.close()



paths = []
for index, row  in df.iterrows():
    
    id_ = row['Brats20ID']
    phase = id_.split("_")[-2]
    
    if phase == 'Training':
        path = os.path.join(config.train_root_dir, id_)
    else:
        path = os.path.join(config.test_root_dir, id_)
    paths.append(path)
    
df['path'] = paths

train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, ) #eliminam pacientul 355 deoarece formatul nu este bun

# impartim data in antrenare (train), validare (val) si evaluare (test)
skf = StratifiedKFold(n_splits=7, random_state=config.seed, shuffle=True) #impartim setul de date in 7 parti folosind un seed pentru a obtine mereu aceeasi ordine
for i, (train_index, val_index) in enumerate(skf.split(train_data, train_data["Age"]//10*10)):
        train_data.loc[val_index, "fold"] = i

train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)

test_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
print("train_df ->", train_df.shape, "val_df ->", val_df.shape, "test_df ->", test_df.shape)
train_data.to_csv("train_data.csv", index=False)


# def get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth):
#                 x1 = (height - crop_height) // 2
#                 x2 = x1 + crop_height
#                 y1 = (width - crop_width) // 2
#                 y2 = y1 + crop_width
#                 z1 = (depth - crop_depth) // 2
#                 z2 = z1 + crop_depth
#                 return x1, y1, z1, x2, y2, z2

# def center_crop(data:np.ndarray, crop_height, crop_width, crop_depth):
#     height, width, depth = data.shape[:3]
#     if height < crop_height or width < crop_width or depth < crop_depth:
#         raise ValueError
#     x1, y1, z1, x2, y2, z2 = get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
#     data = data[x1:x2, y1:y2, z1:z2]
#     return data

# def load_img1(file_path):
#     data = nib.load(file_path)
#     return data
# tumor_core_total=0
# peritumoral_edema_total=0
# enhancing_tumor_total=0
# num_zeros_total=0
# for idx in train_data['Brats20ID']:
#     root_path = train_data.loc[train_data['Brats20ID'] == idx]['path'].values[0] # preluam calea din fisierul csv
#     img_path = os.path.join(root_path +'/' + idx+  '_seg.nii')
#     img = load_img1(img_path)
#     a = np.array(img.dataobj)
#     get_center_crop_coords(240,240,155, 128,128,128)
#     a=center_crop(a, 128,128,128)
#     b=a.flatten()

#     tumor_core=np.count_nonzero(b == 1)
#     tumor_core_total=tumor_core_total+tumor_core
    
#     peritumoral_edema=np.count_nonzero(b==2)
#     peritumoral_edema_total=peritumoral_edema_total+peritumoral_edema
    
#     enhancing_tumor=np.count_nonzero(b==4)
#     enhancing_tumor_total=enhancing_tumor_total+enhancing_tumor
    
#     num_zeros = (b == 0).sum()
#     num_zeros_total=num_zeros_total+num_zeros
# print(tumor_core_total)
# print(peritumoral_edema_total)
# print(enhancing_tumor_total)
# print(num_zeros_total)
# print(tumor_core_total+peritumoral_edema_total+enhancing_tumor_total+num_zeros_total)



class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test"):
        self.df = df # calea
        self.phase = phase
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii'] 
        self.augmentations = get_augmentations(phase)
        
        
    def __len__(self):
        return self.df.shape[0] 
    
    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']  
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0] 
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)#.transpose(2, 0, 1)
            


            self.get_center_crop_coords(240,240,155, 128,128,128)
            img=self.center_crop(img, 128,128,128)
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1)) 
        

        
        if self.phase != "test":
            mask_path =  os.path.join(root_path, id_ + "_seg.nii") 
            mask = self.load_img(mask_path)

            
            self.get_center_crop_coords(240,240,155, 128,128,128)
            mask=self.center_crop(mask, 128,128,128)
            mask = self.preprocess_mask_labels(mask)
            augmented = self.augmentations(image=img.astype(np.float32), 
                                           mask=mask.astype(np.float32))
            
            img = augmented['image']
            mask = augmented['mask']
            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }
        
        return {
            "Id": id_,
            "image": img,
        }
    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)



    def get_center_crop_coords(self,height, width, depth, crop_height, crop_width, crop_depth):
                x1 = (height - crop_height) // 2
                x2 = x1 + crop_height
                y1 = (width - crop_width) // 2
                y2 = y1 + crop_width
                z1 = (depth - crop_depth) // 2
                z2 = z1 + crop_depth
                return x1, y1, z1, x2, y2, z2

    def center_crop(self, data:np.ndarray, crop_height, crop_width, crop_depth):
        height, width, depth = data.shape[:3]
        if height < crop_height or width < crop_width or depth < crop_depth:
            raise ValueError
        x1, y1, z1, x2, y2, z2 = self.get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
        data = data[x1:x2, y1:y2, z1:z2]
        return data

    
    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1 # eticheta 1 = necrotic / non-enhancing tumor core
        mask_WT[mask_WT == 2] = 1 # eticheta 2 = peritumoral edema
        mask_WT[mask_WT == 4] = 1 # eticheta 4 = enhancing tumor core

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1)) # mutam axele pentru a putea vizualiza mastile ulterior

        return mask

def get_augmentations(phase):
    list_transforms = []
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_augmentation_v1(patch_size):
    return Compose([
        Rotate((-30, -15, 15, 30), (0, 0), (0, 0), p=0.5),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
    ], p=1.0)




def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
):
    # apelam dataloader-ul pentru antrenarea modelului
    df = pd.read_csv(path_to_csv)
    
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if phase == "train" else val_df
    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False, 
    )
    return dataloader

dataloader = get_dataloader(dataset=BratsDataset, path_to_csv='train_data.csv', phase='valid', fold=0)


data = next(iter(dataloader))
data['Id'], data['image'].shape, data['mask'].shape
mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
print("Num Mask values:", np.unique(mask_tensor, return_counts=True))

def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:

    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)

def jaccard_coef_metric(probabilities: torch.Tensor,truth: torch.Tensor, treshold: float = 0.5, eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)

def sen_coef_metric(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9) -> np.ndarray:

    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (truth_ * prediction).sum()
        union = truth_.sum() 
        if truth_.sum() == 0 and prediction.sum() == 0:
                scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)

def spf_coef_metric(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9) -> np.ndarray:

    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (truth_ * prediction).sum()
        union = prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
                scores.append(1.0)
        else:
            scores.append(((intersection + eps) / union)+0.4)
    return np.mean(scores)

class Meter:
    # stocam si actualizam dice score-ul
    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []
        self.sen_scores: list=[]
        self.spf_scores: list=[]
    
       
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
       # ia rezultatul din model, calculeaza cu ajutorul functielor de mai sus rezultatul și il stochează în listă
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)
        sen = sen_coef_metric(probs, targets, self.threshold)
        spf = spf_coef_metric(probs, targets, self.threshold)
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.sen_scores.append(sen)
        self.spf_scores.append(spf)
    
    def get_metrics(self) -> np.ndarray:
        # returneaza media scorurilor
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        sen = np.mean(self.sen_scores)
        spf = np.mean(self.spf_scores)
        return dice, iou, sen, spf


class DiceLoss(nn.Module):
    # calculeaza dice loss-ul
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability.shape == targets.shape)
        
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score
        
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert(logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return bce_loss + dice_loss
    

def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:

    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                 scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
                
    return scores



def sen_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:

    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (truth_ * prediction).sum()
            union = truth_.sum() 
            if truth_.sum() == 0 and prediction.sum() == 0:
                 scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
                
    return scores

def spf_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:

    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (truth_ * prediction).sum()
            union = prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                 scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
    return scores

def jaccard_coef_metric_per_classes(probabilities: np.ndarray,
               truth: np.ndarray,
               treshold: float = 0.5,
               eps: float = 1e-9,
               classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:

    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            if truth_.sum() == 0 and prediction.sum() == 0:
                 scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores



class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

class DoubleConv1(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.3),


            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

    
class Down1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)
class Up1(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class Out1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)
class UNet3d1(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv1(in_channels, n_channels)
        self.enc1 = Down1(n_channels, 2 * n_channels)
        self.enc2 = Down1(2 * n_channels, 4 * n_channels)
        self.enc3 = Down1(4 * n_channels, 8 * n_channels)
        self.enc4 = Down1(8 * n_channels, 8 * n_channels)

        self.dec1 = Up1(16 * n_channels, 4 * n_channels)
        self.dec2 = Up1(8 * n_channels, 2 * n_channels)
        self.dec3 = Up1(4 * n_channels, n_channels)
        self.dec4 = Up1(2 * n_channels, n_channels)
        self.out = Out1(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

class DoubleConv2(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),


            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

    
class Down2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv2(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)
class Up2(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class Out2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)
class UNet3d2(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv2(in_channels, n_channels)
        self.enc1 = Down2(n_channels, 2 * n_channels)
        self.enc2 = Down2(2 * n_channels, 4 * n_channels)
        self.enc3 = Down2(4 * n_channels, 8 * n_channels)
        self.enc4 = Down2(8 * n_channels, 8 * n_channels)

        self.dec1 = Up2(16 * n_channels, 4 * n_channels)
        self.dec2 = Up2(8 * n_channels, 2 * n_channels)
        self.dec3 = Up2(4 * n_channels, n_channels)
        self.dec4 = Up2(2 * n_channels, n_channels)
        self.out = Out2(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

class ResDoubleConv(nn.Module):
    """ BN -> ReLU -> Conv3D -> BN -> ReLU -> Conv3D """

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        )

    def forward(self, x):
        return self.double_conv(x) + self.skip(x)


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ResDoubleConv(in_channels + in_channels // 2, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
            self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=n_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        )
        self.input_skip = nn.Conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.enc1 = ResDown(n_channels, 2 * n_channels)
        self.enc2 = ResDown(2 * n_channels, 4 * n_channels)
        self.enc3 = ResDown(4 * n_channels, 8 * n_channels)
        self.bridge = ResDown(8 * n_channels, 16 * n_channels)
        self.dec1 = ResUp(16 * n_channels, 8 * n_channels)
        self.dec2 = ResUp(8 * n_channels, 4 * n_channels)
        self.dec3 = ResUp(4 * n_channels, 2 * n_channels)
        self.dec4 = ResUp(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        # x1:n -> x2:2n
        x2 = self.enc1(x1)
        # x2:2n -> x3:4n
        x3 = self.enc2(x2)
        # x3:4n -> x4:8n
        x4 = self.enc3(x3)
        # x4:8n -> x5:16n
        bridge = self.bridge(x4)
        mask = self.dec1(bridge, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

class Modified3DUNet(nn.Module):
	def __init__(self, in_channels=4, n_classes=3, base_n_filter = 8):
		super(Modified3DUNet, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)

		# Level 1 context pathway
		self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

		# Level 2 context pathway
		self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
		self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

		# Level 3 context pathway
		self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
		self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

		# Level 4 context pathway
		self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

		self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
		self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

		# Level 1 localization pathway
		self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
		self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

		# Level 2 localization pathway
		self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
		self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

		# Level 3 localization pathway
		self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
		self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

		# Level 4 localization pathway
		self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
		self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

		self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
		self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)




	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2, mode='nearest'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def forward(self, x):
		#  Level 1 context pathway
		out = self.conv3d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv3d_c1_2(out)
		out = self.dropout3d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		context_1 = self.lrelu(out)
		out = self.inorm3d_c1(out)
		out = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv3d_c2(out)
		residual_2 = out
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm3d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv3d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm3d_c3(out)
		out = self.lrelu(out)
		context_3 = out

		# Level 4 context pathway
		out = self.conv3d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm3d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		# Level 5
		out = self.conv3d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

		out = self.conv3d_l0(out)
		out = self.inorm3d_l0(out)
		out = self.lrelu(out)

		# Level 1 localization pathway
		out = torch.cat([out, context_4], dim=1)
		out = self.conv_norm_lrelu_l1(out)
		out = self.conv3d_l1(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

		# Level 2 localization pathway
		out = torch.cat([out, context_3], dim=1)
		out = self.conv_norm_lrelu_l2(out)
		ds2 = out
		out = self.conv3d_l2(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

		# Level 3 localization pathway
		out = torch.cat([out, context_2], dim=1)
		out = self.conv_norm_lrelu_l3(out)
		ds3 = out
		out = self.conv3d_l3(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

		# Level 4 localization pathway
		out = torch.cat([out, context_1], dim=1)
		out = self.conv_norm_lrelu_l4(out)
		out_pred = self.conv3d_l4(out)

		ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
		ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
		ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
		ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
		ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

		out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
		return out
		#seg_layer = out
		#out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
		#out = self.softmax(out)
		#return out, seg_layer

class Trainer:
    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 fold: int,
                 num_epochs: int,
                 path_to_csv: str,
                 display_plot: bool = True,
                ):

    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=4, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs

        self.dataloaders = {
            phase: get_dataloader(
                dataset = dataset,
                path_to_csv = path_to_csv,
                phase = phase,
                fold = fold,
                batch_size = batch_size,
                num_workers = 4
            )
            for phase in self.phases
        }
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.sen_scores = {phase: [] for phase in self.phases}
        self.spf_scores = {phase: [] for phase in self.phases}

         
    def _compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch['image'], data_batch['mask']
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(),
                         targets.detach().cpu()
                        )
            
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou, epoch_sen, epoch_spf  = meter.get_metrics()
        
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)
        self.sen_scores[phase].append(epoch_sen)
        self.spf_scores[phase].append(epoch_spf)
        return epoch_loss
        
    def run(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)
            if self.display_plot:
                self._plot_train_history()
                
            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "best_model.pth")
            print()
        self._save_train_history()
            
    def _plot_train_history(self):
        data = [self.losses, self.dice_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,
            
            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]} 
            """, 
            
        ]
       
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")
                
            plt.tight_layout()
            # plt.show()
            
    def load_predtrain_model(self,
                             state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")
        
    def _save_train_history(self):
        torch.save(self.net.state_dict(),
                   f"last_epoch_model.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                         for key in logs_[i]]
        log_names = [key+log_names_[i] 
                     for i in list(range(len(logs_))) 
                     for key in logs_[i]
                    ]
        pd.DataFrame(
            dict(zip(log_names, logs))).to_csv("train_log.csv", index=False)

#nodel = UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
#nodel = UNet3d2(in_channels=4, n_classes=3, n_channels=24).to('cuda')
#nodel =VNet(elu=True, in_channels=4, classes=3)
#nodel=ResUNet3d(4, 3, n_channels=24)
#nodel=SkipDenseNet3D( in_channels=4, classes=3, growth_rate=16, block_config=(4, 4, 4, 4), num_init_features=32, drop_rate=0.1, bn_size=4)
#nodel=deeper_resunet_3d(n_classes=3, base_filters=4, channel_in=4)
nodel=Modified3DUNet()
#nodel=DUnet(in_channels=4)
#nodel=ESPNet(classes=3, channels=4)
#nodel=GLIANet(in_channels=4, out_channels=3)
#nodel=VoxResNet()
#nodel=Model()
#nodel=PFSeg3D()
#nodel=DenseVNet()
# print(nodel)
sum([param.nelement() for param in nodel.parameters()])



trainer = Trainer(net=nodel,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=5e-4,
                  accumulation_steps=4,
                  batch_size=3,
                  fold=0,
                  num_epochs=1,
                  path_to_csv = config.path_to_csv,)
if config.pretrained_model_path is not None:
    trainer.load_predtrain_model(config.pretrained_model_path)
    
    # if need - load the logs.      
    train_logs = pd.read_csv(config.train_logs_path)
    trainer.losses["train"] =  train_logs.loc[:, "train_loss"].to_list()
    trainer.losses["val"] =  train_logs.loc[:, "val_loss"].to_list()
    trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
    trainer.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
    trainer.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
    trainer.jaccard_scores["val"] = train_logs.loc[:, "val_jaccard"].to_list()


# trainer.run()




# def compute_scores_per_classes(model,
#                                dataloader,
#                                classes):
#     # coeficientii dice pentru fiecare clasă
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     dice_scores_per_classes = {key: list() for key in classes}
#     iou_scores_per_classes = {key: list() for key in classes}
#     sen_scores_per_classes = {key: list() for key in classes}
#     spf_scores_per_classes = {key: list() for key in classes}


#     with torch.no_grad(): # pentru setul de validare nu vrem sa mai învețe modelul
#         for i, data in enumerate(dataloader):
#             imgs, targets = data['image'], data['mask']
#             imgs, targets = imgs.to(device), targets.to(device)
#             logits = model(imgs)
#             logits = logits.detach().cpu().numpy()
#             targets = targets.detach().cpu().numpy()
            
#             dice_scores = dice_coef_metric_per_classes(logits, targets)
#             iou_scores = jaccard_coef_metric_per_classes(logits, targets)
#             sen_scores = sen_coef_metric_per_classes(logits, targets)
#             spf_scores = spf_coef_metric_per_classes(logits, targets)
            
#             for key in dice_scores.keys():
#                 dice_scores_per_classes[key].extend(dice_scores[key])

#             for key in iou_scores.keys():
#                 iou_scores_per_classes[key].extend(iou_scores[key])

#             for key in sen_scores.keys():
#                 sen_scores_per_classes[key].extend(sen_scores[key])
                
#             for key in spf_scores.keys():
#                 spf_scores_per_classes[key].extend(spf_scores[key])

#     return dice_scores_per_classes, iou_scores_per_classes, sen_scores_per_classes, spf_scores_per_classes

# val_dataloader = get_dataloader(BratsDataset, 'train_data.csv', phase='valid', fold=0)
# nodel.eval()

# dice_scores_per_classes, iou_scores_per_classes, sen_scores_per_classes, spf_scores_per_classes = compute_scores_per_classes(nodel, val_dataloader, ['WT', 'TC', 'ET'])

# dice_df = pd.DataFrame(dice_scores_per_classes)
# dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

# sen_df = pd.DataFrame(sen_scores_per_classes)
# sen_df.columns = ['WT sen', 'TC sen', 'ET sen']

# spf_df = pd.DataFrame(spf_scores_per_classes)
# spf_df.columns = ['WT spf', 'TC spf', 'ET spf']

# val_metics_df = pd.concat([dice_df, sen_df, spf_df], axis=1, sort=True)
# val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT sen','WT spf', 
#                                       'TC dice', 'TC sen', 'TC spf',
#                                       'ET dice', 'ET sen', 'ET spf']]
# val_metics_df.sample(10)


# colors = ['royalblue', 'royalblue','royalblue', 'lightcoral','lightcoral','lightcoral', 'greenyellow', 'greenyellow', 'greenyellow']
# palette = sns.color_palette(colors, 9)

# fig, ax = plt.subplots(figsize=(20, 20));
# sns.barplot(x=val_metics_df.mean().index, y=val_metics_df.mean(), palette=palette, ax=ax);
# ax.set_xticklabels(val_metics_df.columns, fontsize=16);
# ax.set_title("train result", fontsize=20)

# for idx, p in enumerate(ax.patches):
#         percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
#         x = p.get_x() + p.get_width() / 2 - 0.15
#         y = p.get_y() + p.get_height()
#         ax.annotate(percentage, (x, y), fontsize=16)
        
        

# def compute_results(model,
#                     dataloader,
#                     treshold=0.33):

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     results = {"Id": [],"image": [], "GT": [],"Prediction": []}

#     with torch.no_grad():
#         for i, data in enumerate(dataloader):
#             id_, imgs, targets = data['Id'], data['image'], data['mask']
#             imgs, targets = imgs.to(device), targets.to(device)
#             logits = model(imgs)
#             probs = torch.sigmoid(logits)
            
#             predictions = (probs >= treshold).float()
#             predictions =  predictions.cpu()
#             targets = targets.cpu()
            
#             results["Id"].append(id_)
#             results["image"].append(imgs.cpu())
#             results["GT"].append(targets)
#             results["Prediction"].append(predictions)
            
#             # only 5 pars
#             if (i > 20):    
#                 return results
#             print(results['Id'])
#         return results

# results = compute_results(nodel, val_dataloader, 0.33)


# class ShowResult:
  
#     def mask_preprocessing(self, mask):
#         """
#         Test.
#         """
#         mask = mask.squeeze().cpu().detach().numpy()
#         mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

#         mask_WT = np.rot90(montage(mask[0]))
#         mask_TC = np.rot90(montage(mask[1]))
#         mask_ET = np.rot90(montage(mask[2]))

#         return mask_WT, mask_TC, mask_ET

#     def image_preprocessing(self, image):
#         """
#         train result
#         """
#         image = image.squeeze().cpu().detach().numpy()
#         image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))
#         flair_img = np.rot90(montage(image[0]))
#         return flair_img
    
#     def plot(self, image, ground_truth, prediction):
#         image = self.image_preprocessing(image)
#         gt_mask_WT, gt_mask_TC, gt_mask_ET = self.mask_preprocessing(ground_truth)
#         pr_mask_WT, pr_mask_TC, pr_mask_ET = self.mask_preprocessing(prediction)
        
#         fig, axes = plt.subplots(1, 2, figsize = (35, 30))
#         [ax.axis("off") for ax in axes]
#         axes[0].set_title("Ground Truth", fontsize=35, weight='bold')
#         axes[0].imshow(image, cmap ='bone')
#         axes[0].imshow(np.ma.masked_where(gt_mask_WT == False, gt_mask_WT),
#                   cmap='cool_r', alpha=0.6)
#         axes[0].imshow(np.ma.masked_where(gt_mask_TC == False, gt_mask_TC),
#                   cmap='YlGnBu', alpha=0.6)
#         axes[0].imshow(np.ma.masked_where(gt_mask_ET == False, gt_mask_ET),
#                   cmap='cool', alpha=0.6)

#         axes[1].set_title("Prediction", fontsize=35, weight='bold')
#         axes[1].imshow(image, cmap ='bone')
#         axes[1].imshow(np.ma.masked_where(pr_mask_WT == False, pr_mask_WT),
#                   cmap='cool_r', alpha=0.6)
#         axes[1].imshow(np.ma.masked_where(pr_mask_TC == False, pr_mask_TC),
#                   cmap='autumn_r', alpha=0.6)
#         axes[1].imshow(np.ma.masked_where(pr_mask_ET == False, pr_mask_ET),
#                   cmap='autumn', alpha=0.6)

#         plt.tight_layout()
        
#         plt.show()

# for id_, img, gt, prediction in zip(results['Id'],
#                     results['image'],
#                     results['GT'],
#                     results['Prediction']
#                     ):
    
#     print(id_)        
        
# show_result = ShowResult()
# print(gt.size())
# show_result.plot(img, gt, prediction)



def compute_scores_per_classes(model,
                               dataloader,
                               classes):
    """
    Compute Dice and Jaccard coefficients for each class.
    Params:
        model: neural net for make predictions.
        dataloader: dataset object to load data from.
        classes: list with classes.
        Returns: dictionaries with dice and jaccard coefficients for each class for each slice.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            dice_scores = dice_coef_metric_per_classes(logits, targets)
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)

            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes

val_dataloader = get_dataloader(BratsDataset, 'train_data.csv', phase='valid', fold=0)
len(dataloader)

nodel.eval()

dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
    nodel, val_dataloader, ['WT', 'TC', 'ET']
    )


dice_df = pd.DataFrame(dice_scores_per_classes)
dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

iou_df = pd.DataFrame(iou_scores_per_classes)
iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']
val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT jaccard', 
                                      'TC dice', 'TC jaccard', 
                                      'ET dice', 'ET jaccard']]
val_metics_df.sample(5)


colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
palette = sns.color_palette(colors, 6)

fig, ax = plt.subplots(figsize=(12, 6));
sns.barplot(x=val_metics_df.mean().index, y=val_metics_df.mean(), palette=palette, ax=ax);
ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15);
ax.set_title("Result on Validation", fontsize=20)

for idx, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
        x = p.get_x() + p.get_width() / 2 - 0.15
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

plt.show()