from config import *
from train import *

class plot:
    
    def __init__(self, dice_scores_per_classes, iou_scores_per_classes):

        dice_df = pd.DataFrame(dice_scores_per_classes)
        dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

        iou_df = pd.DataFrame(iou_scores_per_classes)
        iou_df.columns = ['WT IoU', 'TC IoU', 'ET IoU']
        
        print(dice_df.head(10))
        # val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
        # val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT IoU', 
        #                                     'TC dice', 'TC IoU', 
        #                                     'ET dice', 'ET IoU']]

        # colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
        # palette = sns.color_palette(colors, 6)

        # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

        # for idx, col in enumerate(val_metics_df.columns):
        #     row = idx // 3
        #     col = idx % 3
        #     sns.histplot(data=val_metics_df[col], ax=axes[row, col], color=palette[idx])
        #     axes[row, col].set_title(col)
        #     axes[row, col].set_xlabel('Score')
        #     axes[row, col].set_ylabel('Frequency')

        # plt.tight_layout()
        # plt.show()