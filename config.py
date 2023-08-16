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

import albumentations as A
from albumentations.pytorch import ToTensorV2 
from albumentations import HorizontalFlip, VerticalFlip, Normalize, Compose

import warnings
warnings.simplefilter("ignore")


class configuration:
    '''in this case we devide trainingData into valid set and train set and test set,
     also the ValidationData considers as test set for model evaluation'''

    train_path = '/home/maximum/Desktop/tf2/dataSet/MICCAI_BraTS2020_TrainingData'
    valid_path = '/home/maximum/Desktop/tf2/dataSet/MICCAI_BraTS2020_ValidationData'
    train_csv_path = 'trainResult/train_data.csv'
    pretrained_model_path = None # '/home/maximum/Desktop/tf2/torch/pre/best_model.pth'
    train_logs_path = None #'/home/maximum/Desktop/tf2/torch/pre/train_log.csv'
    seed = 55
    
def random_seed(seed: int):
    '''set random seed for initializing'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)