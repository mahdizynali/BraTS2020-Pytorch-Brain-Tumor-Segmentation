from config import *
from dataGenerator import extractPath, generator

data = extractPath()
train_df, val_df, test_df = data.train_test_valid()
# data.plotResult(train_df, val_df, test_df)

def get_dataloader(dataset: torch.utils.data.Dataset,
                   path_to_csv, mode, fold=0,
                   batch_size=configuration.batch_size,
                   num_workers=configuration.num_workers):
    # apelam dataloader-ul pentru antrenarea modelului
    df = pd.read_csv(path_to_csv)
    
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True) # fold mentions Kfold method
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if mode == "train" else val_df
    dataset = dataset(df, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,   
    )
    return dataloader

dataloader = get_dataloader(dataset=generator, path_to_csv=configuration.train_csv_path, mode='valid', fold=0)

# data = next(iter(dataloader))
# data['Id'], data['image'].shape, data['mask'].shape
# mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
# print("Num Mask values:", np.unique(mask_tensor, return_counts=True))


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

trainer.run()