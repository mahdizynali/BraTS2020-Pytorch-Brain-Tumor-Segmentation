from config import *
from dataGenerator import generator
from model import *
from coefficent import *
from train import Trainer, data_loader
from model import bestUnet
from plotter import *

model = bestUnet()
sum([param.nelement() for param in model.parameters()])

pred = Trainer(net=model,
                  fold=0,
                  dataset=generator,
                  criterion=BCEDiceLoss(),
                  lr=configuration.learnin_rate,
                  accumulation_steps=configuration.acc_steps,
                  batch_size = configuration.batch_size,
                  num_epochs = configuration.epochs,
                  path_to_csv = configuration.train_csv_path)

pred.load_predtrain_model(configuration.pretrained_model_path)
train_logs = pd.read_csv(configuration.train_logs_path)
pred.losses["train"] =  train_logs.loc[:, "train_loss"].to_list()
pred.losses["val"] =  train_logs.loc[:, "valid_loss"].to_list()
pred.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
pred.dice_scores["val"] = train_logs.loc[:, "valid_dice"].to_list()
pred.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
pred.jaccard_scores["val"] = train_logs.loc[:, "valid_jaccard"].to_list()


val_dataloader = data_loader(dataset=generator, path_to_csv=configuration.train_csv_path, phase='valid', fold=0)
model.eval()
dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(model, val_dataloader, ['WT', 'TC', 'ET'])

# test_on_dataset(dice_scores_per_classes, iou_scores_per_classes)
# plot_train_history()
# plot_dice_history_per_class()
# plot_iou_history_per_class()
