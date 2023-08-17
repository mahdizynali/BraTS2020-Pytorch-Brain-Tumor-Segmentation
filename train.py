from config import *
from dataGenerator import extractPath, generator
from model import *
from coefficent import *

data = extractPath()
train_df, val_df, test_df = data.train_test_valid()
# data.plotResult(train_df, val_df, test_df)

def data_loader(dataset: torch.utils.data.Dataset,
                   path_to_csv, phase, fold=0,
                   batch_size=configuration.batch_size,
                   num_workers=configuration.num_workers):
    # apelam dataloader-ul pentru antrenarea modelului
    df = pd.read_csv(path_to_csv)
    
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True) # fold mentions Kfold method
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if phase == "train" else val_df
    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=configuration.batch_size,
        num_workers=configuration.num_workers,
        pin_memory=True,
        shuffle=False,   
    )
    return dataloader

dataloader = data_loader(dataset=generator, path_to_csv=configuration.train_csv_path, phase='valid', fold=0)



class Trainer:
    def __init__(self, net: nn.Module, dataset: torch.utils.data.Dataset, criterion: nn.Module,
                 lr, accumulation_steps, batch_size, fold, num_epochs, path_to_csv, display_plot = True):

        print("Try to start processing ...\n")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            phase: data_loader(
                dataset = dataset,
                path_to_csv = path_to_csv,
                phase = phase,
                fold = fold,
                batch_size = batch_size,
                num_workers = configuration.num_workers
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
        print(f"########## {phase} epoch number {epoch + 1} ##########")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        counter = 0
        for itr, data_batch in enumerate(dataloader):
            print(f"epoch: {epoch + 1} | step: {counter} | time: {time.strftime('%H:%M:%S')}")
            counter += 1
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
        
    def setup(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)
            if self.display_plot:
                self._plot_train_history()
                
            if val_loss < self.best_loss:
                print(f"\n{'#'*40}\nSaved checkpoint epoch {epoch + 1}\n{'#'*40}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "trainResult/best_model.pth")
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
                   f"trainResult/last_epoch_model.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                         for key in logs_[i]]
        log_names = [key+log_names_[i] 
                     for i in list(range(len(logs_))) 
                     for key in logs_[i]
                    ]
        pd.DataFrame(
            dict(zip(log_names, logs))).to_csv("trainResult/train_log.csv", index=False)
        


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



train = Trainer(net=nodel,
                  fold=0,
                  dataset=generator,
                  criterion=BCEDiceLoss(),
                  lr=configuration.learnin_rate,
                  accumulation_steps=configuration.acc_steps,
                  batch_size = configuration.batch_size,
                  num_epochs = configuration.epochs,
                  path_to_csv = configuration.train_csv_path)

train.setup()