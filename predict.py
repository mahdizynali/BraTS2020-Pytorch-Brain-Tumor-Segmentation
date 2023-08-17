from config import *
from dataGenerator import extractPath, generator
from model import *
from coefficent import *
from train import Trainer
from model import Modified3DUNet

model = Modified3DUNet()
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
pred.losses["val"] =  train_logs.loc[:, "val_loss"].to_list()
pred.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
pred.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
pred.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
pred.jaccard_scores["val"] = train_logs.loc[:, "val_jaccard"].to_list()