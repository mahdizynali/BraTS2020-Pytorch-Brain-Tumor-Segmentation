from config import *
from dataGenerator import extractPath, generator

data = extractPath()
train_df, val_df, test_df = data.train_test_valid()
# data.plotResult(train_df, val_df, test_df)

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

dataloader = get_dataloader(dataset=generator, path_to_csv=configuration.train_csv_path, phase='valid', fold=0)

data = next(iter(dataloader))
data['Id'], data['image'].shape, data['mask'].shape
mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
# print("Num Mask values:", np.unique(mask_tensor, return_counts=True))