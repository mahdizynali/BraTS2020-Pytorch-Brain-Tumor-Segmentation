from config import *
from dataGenerator import extractPath, generator
from model import *
from coefficent import *

data = extractPath()
train_df, val_df, test_df = data.train_test_valid()
# data.plotResult(train_df, val_df, test_df)

def data_loader(dataset: torch.utils.data.Dataset,path_to_csv, phase, fold=0):

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

# dataloader = data_loader(dataset=generator, path_to_csv=configuration.train_csv_path, phase='valid', fold=0)

class Trainer:
    def __init__(self, net: nn.Module, dataset: torch.utils.data.Dataset, criterion: nn.Module,
                 lr, accumulation_steps, batch_size, fold, num_epochs, path_to_csv, display_plot = False):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Try to start processing on {self.device} ...\n")
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=4, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "valid"]
        self.num_epochs = num_epochs
        self.hist_class_dice = []
        self.hist_class_iou = []

        self.dataloaders = {
            phase: data_loader(
                dataset = dataset,
                path_to_csv = path_to_csv,
                phase = phase,
                fold = fold,
            )
            for phase in self.phases
        }
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.sen_scores = {phase: [] for phase in self.phases}
        self.spf_scores = {phase: [] for phase in self.phases}

         
    def compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits
    
    def final_train_valid_per_class(self, data):
        mean_dict = {}
        for data_dict in data:
            for key, value in data_dict.items():
                if key not in mean_dict:
                    mean_dict[key] = value[0]
                else:
                    mean_dict[key] += value[0]
               
        num_dicts = len(data)
        for key in mean_dict.keys():
            mean_dict[key] /= num_dicts

        return mean_dict

    def do_epoch(self, epoch: int, phase: str):
        print("#"*50)
        t = time.process_time()
        print(f"---------- {phase} initial mode epoch {epoch + 1} ----------\n")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()

        for itr, data_batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Steps"):   
            dice_per_classes, iou_per_classes = compute_scores_per_classes_batch(self.net, data_batch, ['WT', 'TC', 'ET']) 
            self.hist_class_dice.append(dice_per_classes)
            self.hist_class_iou.append(iou_per_classes)
            images, targets = data_batch['image'], data_batch['mask']
            loss, logits = self.compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(), targets.detach().cpu())
            
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou, epoch_sen, epoch_spf  = meter.get_metrics()
        elapsed_time = time.process_time() - t
        mean_dice_per_class = self.final_train_valid_per_class(self.hist_class_dice)
        mean_iou_per_class = self.final_train_valid_per_class(self.hist_class_iou)

        print(f"loss : {epoch_loss} \ndice : {epoch_dice}\nIoU : {epoch_iou}")
        print(f"epoch time : {elapsed_time:.2f} sec")       
        print(f"Dice per class : {mean_dice_per_class}")
        print(f"IoU per class : {mean_iou_per_class}\n")
        
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)
        self.sen_scores[phase].append(epoch_sen)
        self.spf_scores[phase].append(epoch_spf)
        return epoch_loss
            
    def plot_train_history(self):
        data = [self.losses, self.dice_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['valid'][-1]}
            """,
            
            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['valid'][-1]} 
            """, 
            
        ]
       
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['valid'], c=colors[0], label="valid")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")
                
            plt.tight_layout()
            plt.show()

    def setup(self):
        for epoch in range(self.num_epochs):
            self.do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self.do_epoch(epoch, "valid")
                self.scheduler.step(val_loss)
            if self.display_plot:
                self.plot_train_history()
                
            if val_loss < self.best_loss:
                print(f"\nSaved new best checkpoint epoch {epoch + 1}\n{'#'*50}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "trainResult/best_model.pth")
            print()
        self.save_train_history()
            
    def load_predtrain_model(self, state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrained model loaded\n")
        
    def save_train_history(self):
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
        
def run() :
    nodel=Modified3DUNet()
    sum([param.nelement() for param in nodel.parameters()])
    trainer = Trainer(net=nodel,
                    fold=0,
                    dataset=generator,
                    criterion=BCEDiceLoss(),
                    lr=configuration.learnin_rate,
                    accumulation_steps=configuration.acc_steps,
                    batch_size = configuration.batch_size,
                    num_epochs = configuration.epochs,
                    path_to_csv = configuration.train_csv_path)
    trainer.setup()

run()