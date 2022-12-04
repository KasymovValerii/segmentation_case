import tqdm
import numpy as np
import torch
import wandb
import os

from torch import nn
from torch.utils.data import DataLoader, Dataset

from myUtils import plot_learning_step
from myLoss import DiceLoss

def train(model: nn.Module,
          trainloader: DataLoader,
          testloader: DataLoader,
          testdataset: Dataset,
          optimizer: torch.optim,
          criterion: nn.Module,
          device: str,
          track=False,
          dir_to_save=None,
          num_epochs=20, 
          notebook=False) -> list:
  """
  Тренирует модель.
  Трекер: wandb.
  Возвращает список из DiceLoss с каждой эпохи. 
  ВАЖНО!!!:  каждые 25 эпох батчи с тестовыми данными меняются.
  model: input_channels = 1
  """

  X_val, Y_val = next(iter(testloader))
  losses = []
  dice_list = []
  prev_dice = float('+inf') 
  dir_to_save = os.getcwd() if not dir_to_save else dir_to_save
  dir_to_save += '/outputs.pickle'

  if notebook:
    tqdm_line = tqdm.notebook.tqdm
  else:
    tqdm_line = tqdm.tqdm
    
  for epoch in tqdm_line(range(1, num_epochs + 1)):
    model = model.to(device)
    model.train()

    epoch_losses = []
    for inputs, target in trainloader:
      inputs = inputs.to(device)
      target = target.to(device)

      optimizer.zero_grad()
      prediction = model(inputs)
      loss = criterion(prediction, target)
      loss.backward()

      optimizer.step()

      epoch_losses.append(loss.cpu().detach().numpy())
    
    
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)

    if epoch % 25 == 0:
      X_val, Y_val = next(iter(testloader))
    
    if epoch % 100 == 0:
      optimizer.param_groups[0]['lr'] /= 10
    
    model.eval()
    with torch.no_grad():
      model = model.to('cpu')
      Y_hat = model(X_val).detach()
      test_loss = criterion(Y_hat, Y_val)
      dice_loss = DiceLoss()(Y_hat, Y_val)

    dice_list.append(dice_loss)
    if dice_loss < prev_dice:
      prev_dice = dice_loss
      torch.save(model, dir_to_save)

    Y_hat = torch.sigmoid(Y_hat).numpy()
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1  
    plot_learning_step(val_data=Y_val,
                       pred_data=Y_hat,
                       epoch=epoch,
                       num_epochs=num_epochs,
                       avg_loss=avg_loss,
                       test_loss=test_loss)

    if track:
      wandb.log({'train loss BCE': avg_loss, 'epoch': epoch})
      wandb.log({'test loss BCE': test_loss, 'epoch': epoch})
      wandb.log({'test loss Dice': dice_loss, 'epoch': epoch})
  
  return dice_list
