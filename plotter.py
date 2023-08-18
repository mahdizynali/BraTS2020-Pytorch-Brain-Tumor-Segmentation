from config import *
from train import *
    
def test_on_dataset(dice_scores_per_classes, iou_scores_per_classes):
    dice_df = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

    iou_df = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns = ['WT IoU', 'TC IoU', 'ET IoU']
    
    val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT IoU', 
                                        'TC dice', 'TC IoU', 
                                        'ET dice', 'ET IoU']]

    val_metics_df.sample(20)

    colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    palette = sns.color_palette(colors, 6)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=val_metics_df.mean().index, y=val_metics_df.mean(), palette=palette, ax=ax)
    ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15)
    ax.set_title("Result on Validation", fontsize=20)

    for idx, p in enumerate(ax.patches):
            percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
            x = p.get_x() + p.get_width() / 2 - 0.15
            y = p.get_y() + p.get_height()
            ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    plt.show()

def plot_train_history():
    df = pd.read_csv(configuration.train_log)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["train_dice"], label="Training Dice")
    plt.plot(df["valid_dice"], label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training and Validation Dice Scores")
    plt.legend()

    max_train_dice = df["train_dice"].max()
    max_valid_dice = df["valid_dice"].max()

    plt.axhline(y=max_train_dice, color='r', linestyle='--', alpha=0.3, label=f'Max Train Dice: {max_train_dice:.2f}')
    plt.axhline(y=max_valid_dice, color='g', linestyle='--', alpha=0.3, label=f'Max Valid Dice: {max_valid_dice:.2f}')
    
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["train_loss"], label="Training Loss")
    plt.plot(df["valid_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()

    min_train_loss = df["train_loss"].min()
    min_valid_loss = df["valid_loss"].min()

    plt.axhline(y=min_train_loss, color='r', linestyle='--', alpha=0.3, label=f'Min Train Loss: {min_train_loss:.2f}')
    plt.axhline(y=min_valid_loss, color='g', linestyle='--', alpha=0.3, label=f'Min Valid Loss: {min_valid_loss:.2f}')
    
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/train-result.png")
    plt.show()

def plot_dice_history_per_class():
    df = pd.read_csv(configuration.dice_per_class_path)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["WT-dice-train"], label="Whole Tumor")
    plt.plot(df["TC-dice-train"], label="Tumor Core")
    plt.plot(df["ET-dice-train"], label="Enhanced Tumor")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training Dice Scores")
    plt.legend()

    max_wt_train_dice = df["WT-dice-train"].max()
    max_tc_train_dice = df["TC-dice-train"].max()
    max_et_train_dice = df["ET-dice-train"].max()

    plt.axhline(y=max_wt_train_dice, color='r', linestyle='--', alpha=0.3, label=f'Max WT Dice: {max_wt_train_dice:.2f}')
    plt.axhline(y=max_tc_train_dice, color='g', linestyle='--', alpha=0.3, label=f'Max TC Dice: {max_tc_train_dice:.2f}')
    plt.axhline(y=max_et_train_dice, color='b', linestyle='--', alpha=0.3, label=f'Max ET Dice: {max_et_train_dice:.2f}')
    
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["WT-dice-val"], label="Whole Tumor")
    plt.plot(df["TC-dice-val"], label="Tumor Core")
    plt.plot(df["ET-dice-val"], label="Enhanced Tumor")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Scores")
    plt.legend()

    max_wt_val_dice = df["WT-dice-val"].max()
    max_tc_val_dice = df["TC-dice-val"].max()
    max_et_val_dice = df["ET-dice-val"].max()

    plt.axhline(y=max_wt_val_dice, color='r', linestyle='--', alpha=0.3, label=f'Max WT Dice: {max_wt_val_dice:.2f}')
    plt.axhline(y=max_tc_val_dice, color='g', linestyle='--', alpha=0.3, label=f'Max TC Dice: {max_tc_val_dice:.2f}')
    plt.axhline(y=max_et_val_dice, color='b', linestyle='--', alpha=0.3, label=f'Max ET Dice: {max_et_val_dice:.2f}')
    
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/train-dice-per-class.png")
    plt.show()
    
def plot_iou_history_per_class():
    df = pd.read_csv(configuration.iou_per_class_path)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["WT-IoU-train"], label="Whole Tumor")
    plt.plot(df["TC-IoU-train"], label="Tumor Core")
    plt.plot(df["ET-IoU-train"], label="Enhanced Tumor")
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("Training IoU Scores")
    plt.legend()

    max_wt_train_iou = df["WT-IoU-train"].max()
    max_tc_train_iou = df["TC-IoU-train"].max()
    max_et_train_iou = df["ET-IoU-train"].max()

    plt.axhline(y=max_wt_train_iou, color='r', linestyle='--', alpha=0.3, label=f'Max WT IoU: {max_wt_train_iou:.2f}')
    plt.axhline(y=max_tc_train_iou, color='g', linestyle='--', alpha=0.3, label=f'Max TC IoU: {max_tc_train_iou:.2f}')
    plt.axhline(y=max_et_train_iou, color='b', linestyle='--', alpha=0.3,label=f'Max ET IoU: {max_et_train_iou:.2f}')
    
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["WT-IoU-val"], label="Whole Tumor")
    plt.plot(df["TC-IoU-val"], label="Tumor Core")
    plt.plot(df["ET-IoU-val"], label="Enhanced Tumor")
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("Validation IoU Scores")
    plt.legend()

    max_wt_val_iou = df["WT-IoU-val"].max()
    max_tc_val_iou = df["TC-IoU-val"].max()
    max_et_val_iou = df["ET-IoU-val"].max()

    plt.axhline(y=max_wt_val_iou, color='r', linestyle='--', alpha=0.3, label=f'Max WT IoU: {max_wt_val_iou:.2f}')
    plt.axhline(y=max_tc_val_iou, color='g', linestyle='--', alpha=0.3, label=f'Max TC IoU: {max_tc_val_iou:.2f}')
    plt.axhline(y=max_et_val_iou, color='b', linestyle='--', alpha=0.3, label=f'Max ET IoU: {max_et_val_iou:.2f}')
    
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/train-iou-per-class.png")
    plt.show()
    
    
class ShowResult:
  
    def mask_preprocessing(self, mask):

        mask = mask.squeeze().cpu().detach().numpy()
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        mask_WT = np.rot90(montage(mask[0]))
        mask_TC = np.rot90(montage(mask[1]))
        mask_ET = np.rot90(montage(mask[2]))

        return mask_WT, mask_TC, mask_ET

    def image_preprocessing(self, image):

        image = image.squeeze().cpu().detach().numpy()
        image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))
        flair_img = np.rot90(montage(image[0]))
        return flair_img
    
    def plot(self, image, ground_truth, prediction):
        image = self.image_preprocessing(image)
        gt_mask_WT, gt_mask_TC, gt_mask_ET = self.mask_preprocessing(ground_truth)
        pr_mask_WT, pr_mask_TC, pr_mask_ET = self.mask_preprocessing(prediction)
        
        fig, axes = plt.subplots(1, 2, figsize = (35, 30))
        [ax.axis("off") for ax in axes]
        axes[0].set_title("Ground Truth", fontsize=35, weight='bold')
        axes[0].imshow(image, cmap ='bone')
        axes[0].imshow(np.ma.masked_where(gt_mask_WT == False, gt_mask_WT),
                  cmap='cool_r', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_TC == False, gt_mask_TC),
                  cmap='YlGnBu', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_ET == False, gt_mask_ET),
                  cmap='cool', alpha=0.6)

        axes[1].set_title("Prediction", fontsize=35, weight='bold')
        axes[1].imshow(image, cmap ='bone')
        axes[1].imshow(np.ma.masked_where(pr_mask_WT == False, pr_mask_WT),
                  cmap='cool_r', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_TC == False, pr_mask_TC),
                  cmap='autumn_r', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_ET == False, pr_mask_ET),
                  cmap='autumn', alpha=0.6)

        plt.tight_layout()
        plt.show()