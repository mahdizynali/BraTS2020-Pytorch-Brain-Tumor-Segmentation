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

    plt.subplot(1, 2, 2)
    plt.plot(df["train_loss"], label="Training Loss")
    plt.plot(df["valid_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
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

    plt.subplot(1, 2, 2)
    plt.plot(df["WT-dice-val"], label="Training Loss")
    plt.plot(df["TC-dice-val"], label="Validation Loss")
    plt.plot(df["ET-dice-val"], label="Enhanced Tumor")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Scores")
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
    plt.ylabel("Dice Score")
    plt.title("Training IoU Scores")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["WT-IoU-val"], label="Training Loss")
    plt.plot(df["TC-IoU-val"], label="Validation Loss")
    plt.plot(df["ET-IoU-val"], label="Enhanced Tumor")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation IoU Scores")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/train-IoU-per-class.png")
    plt.show()