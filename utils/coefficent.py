from config import *
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

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

def get_coordinates(mask):
    indices = torch.nonzero(mask)
    return indices.tolist()

def hausdorff_distance(probs, targets, threshold):
    predictions = (probs >= threshold).float()
    hausdorff_distances = []
    for pred_mask, gt_mask in zip(predictions, targets):
        pred_coords = get_coordinates(pred_mask)  
        gt_coords = get_coordinates(gt_mask)      
        distance = max(directed_hausdorff(pred_coords, gt_coords)[0], 
            directed_hausdorff(gt_coords, pred_coords)[0]) / 10
        hausdorff_distances.append(distance)
    hausdorff_95 = np.percentile(hausdorff_distances, 95)
    return hausdorff_95

class DiceLoss(torch.nn.Module):
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
        
class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert(logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return bce_loss + dice_loss

class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, weights):
        super(WeightedCrossEntropy, self).__init__()
        self.class_frequencies = weights
        self.class_weights = 1.0 / np.array(self.class_frequencies)
        self.class_weights /= np.sum(self.class_weights)
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.class_weights = self.class_weights.to(logits.device)
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, weight=self.class_weights)
        return ce_loss
    
def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:

    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    # assert(predictions.shape == truth.shape)

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


def compute_scores_per_classes(model, dataloader, classes):

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

def scale_fusion(self, ds2, ds3, out_pred):
    ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
    ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
    ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
    ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
    ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
    out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
    return out


def compute_scores_per_classes_batch(model, data, classes):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        imgs, targets = data['image'], data['mask']
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        logits = logits.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        dice_scores = dice_coef_metric_per_classes(logits, targets)
        iou_scores = jaccard_coef_metric_per_classes(logits, targets)

        for key in dice_scores.keys():
            mn = sum(dice_scores[key]) / len(dice_scores[key])
            dice_scores_per_classes[key].append(mn)

        for key in iou_scores.keys():
            mn = sum(iou_scores[key]) / len(iou_scores[key])
            iou_scores_per_classes[key].append(mn)

    return dice_scores_per_classes, iou_scores_per_classes
