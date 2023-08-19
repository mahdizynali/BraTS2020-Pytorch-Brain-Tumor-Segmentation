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




modality_types = ['flair', 't1', 't1ce', 't2']
# def show_results (model, test_dl, checkpoint_path = None):
#     """ showing image, mask and predicted mask for one batch """
#     if checkpoint_path is not None:
#         load_checkpoint (torch.load (checkpoint_path), model)
#     dl = iter(test_dl)
# #     dl.next()
# #     dl.next()
# #     dl.next()
# #     dl.next()
# #     dl.next()
# #     dl.next()
# #     dl.next()
#     images, masks = dl.next()

#     for BB in range(4):
#         images, masks = dl.next()
#         images = images.to("cpu")
#         masks = masks.to("cpu")
#         outputs = model (images.float())

#         preds = torch.argmax (outputs, dim = 1)
#         masks = torch.argmax (masks, dim = 1)
# #         print(torch.unique(preds), torch.unique(masks))
#         masks = masks*84
#         preds = preds*84

#         mean = 0.5
#         std = 0.5
#         plt.figure (figsize = (20, 40))
#         for i in range (8):
#             for j in range (len (modality_types)):
#                 # show all type of images
#                 plt.subplot (16, 6, 6 * i + j + 1)
#                 plt.axis ('off')
#                 plt.title (modality_types [j])
#     #             image = gpu_to_cpu (images [i][j], std, mean)
#                 plt.imshow (images [i][j].cpu(), cmap = 'bone')
#             # show True Mask
#             plt.subplot (16, 6, 6 * i + 5)
#             plt.title ('True Mask')
#             plt.axis ('off')
#             plt.imshow (255 - masks[i].cpu(), cmap = 'bone')
#             # show Predicted Mask
#             plt.subplot (16, 6, 6 * i + 6)
#             plt.title ('Predicted Mask')
#             plt.axis ('off')
#     #         pred = gpu_to_cpu (preds [i], std, mean)
#             plt.imshow (255 - preds[i].cpu(), cmap = 'bone')

#         plt.show ()
#     return masks, preds
# show_results (model, valid_dl, checkpoint_path = None)





# train_ds = generator(train_dirs, modality_types)
# valid_ds = generator(valid_dirs, modality_types)
# train_dl = DataLoader(train_ds, batch_size = 8, shuffle = False, num_workers = 2, pin_memory = True)
# valid_dl = DataLoader(valid_ds, batch_size = 8, shuffle = False, num_workers = 2, pin_memory = True)

# dl_it=iter (train_dl)
# dl_it.next ()
# dl_it.next ()
# dl_it.next ()
# dl_it.next ()
# imgs, msks = dl_it.next ()
# print (imgs.shape)
# print (msks.shape)

# idx = np.random.permutation (imgs.shape [0])
# imgs = imgs [idx]
# msks = msks [idx]

# plt.figure (figsize = (25, 20))
# for i in range (4):
#     for j in range (len (modality_types)):
#         plt.subplot (4, 5, 5 * i + j + 1)
#         plt.imshow (imgs[i][j], cmap = 'bone')
#         plt.axis ('off')
#         plt.title (modality_types [j])
#     plt.subplot (4, 5, 5 * i + 5)
#     plt.imshow (256-(np.argmax(msks[i], axis=0)*80 ), cmap='gray')
#     plt.axis ('off')
#     plt.title ('Mask')

