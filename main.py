import myUtils
import myDataset
import myModel
import myLoss
from myTrain import train

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import os
import numpy as np
import wandb
import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def solution(main_dir=None, track=False) -> None:
  main_dir = os.getcwd() if not main_dir else main_dir
  dir_images = main_dir + "/subset/subset"
  dir_masks = main_dir + "/subset/subset_masks"

  data_dict = myUtils.read_images(dir_images)
  masks_dict = myUtils.read_masks(dir_masks)

  img_tensor, msk_tensor = myUtils.dicts_to_tensors(data_dict, masks_dict)

  X_train, X_test, y_train, y_test = train_test_split(img_tensor, msk_tensor, train_size=0.7, shuffle=True)

  traindataset = myDataset.TorchData(X_train[:, np.newaxis, :, :], y_train[:, np.newaxis, :, :])
  testdataset = myDataset.TorchData(X_test[:, np.newaxis, :, :], y_test[:, np.newaxis, :, :])

  trainloader = DataLoader(traindataset,
                          batch_size=16,
                          num_workers=True,
                          shuffle=True,
                          drop_last=True)
  testloader = DataLoader(traindataset,
                          batch_size=6,
                          num_workers=True,
                          shuffle=True,
                          drop_last=True)
  if track:
    wandb.login()
    wandb.init(
        project="test_task", 
        config={
        "architecture": "Net enc-batchnorm",
        "dataset": "Tomograms",
        "loss": "dice loss",
        "epochs": 200,
        "time": datetime.datetime.now().strftime("%H:%M:%S")
        })

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model = myModel.Net()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.BCEWithLogitsLoss()

  dice_loss = train(model=model,
                      trainloader=trainloader,
                      testloader=testloader,
                      testdataset=testdataset,
                      optimizer=optimizer,
                      criterion=criterion,
                      device=device,
                      track=track,
                      dir_to_save=main_dir,
                      num_epochs=200, 
                      notebook=True)
  if track:
    wandb.finish()

  epochs = [i + 1 for i in range(len(dice_loss))]
  fig, ax = plt.subplots()
  ax.plot(dice_loss)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Dice Loss')
  ax.grid(which='major')
  ax.set_xticklabels(epochs[1::2])

  plt.show()

  fig.savefig(main_dir + '/DiceLoss.png')


if __name__ == '__main__':
  solution()
            





