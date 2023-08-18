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
plot_train_history()
plot_dice_history_per_class()
plot_iou_history_per_class()


# def compute_results(model, dataloader, treshold=0.50):

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     results = {"Id": [],"image": [], "GT": [],"Prediction": []}

#     with torch.no_grad():
#         for i, data in enumerate(dataloader):
#             id_, imgs, targets = data['Id'], data['image'], data['mask']
#             imgs, targets = imgs.to(device), targets.to(device)
#             logits = model(imgs)
#             probs = torch.sigmoid(logits)
            
#             predictions = (probs >= treshold).float()
#             predictions =  predictions.cpu()
#             targets = targets.cpu()
            
#             results["Id"].append(id_)
#             results["image"].append(imgs.cpu())
#             results["GT"].append(targets)
#             results["Prediction"].append(predictions)
            
#             # only 5 pars
#             if (i > 20):    
#                 return results
#             print(results['Id'])
#         return results

# results = compute_results(model, val_dataloader, 0.40)
# for id_, img, gt, prediction in zip(results['Id'], results['image'], results['GT'], results['Prediction']):
#     print(id_)
# show_result = ShowResult()
# show_result.plot(img, gt, prediction)






# device =  'cpu'
# with torch.no_grad():
#     images = data['image'][tio.DATA].to(device)
#     targets = data['label'][tio.DATA].to(device)
#     predict = model(images)
#     img_tensor = data['image'][tio.DATA].squeeze()[1].cpu().detach().numpy() 
#     mask_tensor = data['label'][tio.DATA].squeeze()[0].squeeze().cpu().detach().numpy()
#     predict_tensor = predict.squeeze()[0].squeeze().cpu().detach().numpy()
#     image = np.rot90(montage(img_tensor))
#     mask_predict = np.rot90(montage(predict_tensor))
#     mask = np.rot90(montage(mask_tensor)) 
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 20))
#     ax1.imshow(image, cmap = 'gray')
#     ax1.imshow(np.ma.masked_where(mask == False, mask),
#     cmap='cool', alpha=0.6)
#     ax2.imshow(image, cmap ='gray')
#     ax2.imshow(np.ma.masked_where(mask == False, mask_predict),
#     cmap='cool', alpha=0.6)

