import torch
import numpy as np

import warnings
warnings.simplefilter("ignore")

class configuration:
    '''in this case we devide trainingData into valid set and train set and test set,
     also the ValidationData considers as test data after training'''

    train_path = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_TrainingData'
    test_path = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_ValidationData'
    train_csv_path = 'trainResult/train_data.csv'
    train_log = 'trainResult/train_log.csv'
    dice_per_class_path = 'trainResult/dice_perـclass.csv'
    iou_per_class_path = 'trainResult/iou_per_class.csv'
    pretrained_model_path = "trainResult/best_model.pth"
    train_logs_path = "trainResult/train_log.csv"
    seed = 55
    batch_size = 4
    epochs = 1
    learnin_rate = 5e-4
    num_workers = 4
    acc_steps = 4
    
def random_seed(seed: int):
    '''set random seed for initializing'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)