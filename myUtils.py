import SimpleITK as sitk
import numpy as np
import copy
import os
import scipy.ndimage
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from typing import Dict, Tuple
from IPython.display import clear_output

def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx


def search_dir_with_files(directory: str) -> str:
  """
  Погружение в глубину папок для возвращения папки с файлами.
  ВНИМАНИЕ!!! Работает только со структурой директорий в задании.
  """
  dir = copy.copy(directory)
  while len(os.listdir(dir)) == 1:
    dir = dir + '/' + os.listdir(dir)[0]
  folder = [s for s in os.listdir(dir) if not s.endswith('.json')][0]
  return dir + '/' + folder


def read_images(dir: str) -> Dict:
  """
  Проходит по всем папкам с изображениями и считывает данные.
  data_dict: {Имя папки: Данные}
  Данные: 3-ех мерный массив с размером (количество данных, размер картинки 1, размер картинки 2)
  """
  data_dict = {}
  i = 0
  list_dir = sorted(os.listdir(dir))
  for folder in list_dir:
    data_dir = search_dir_with_files(dir + '/' + folder)
    data_dict[folder] = load_dicom(data_dir)
    if i == 2:
      pass
    i += 1
  return data_dict


def read_masks(dir: str) -> Dict:
  """
  Читает маски, записывает в словарь
  """
  masks_dict = {}
  i = 0
  list_dir = sorted(os.listdir(dir))
  for folder in list_dir:
    directory = dir + '/' + folder
    filename = directory + '/' + os.listdir(directory)[0]
    mask = nib.load(filename)
    mask = mask.get_fdata().transpose(2, 0, 1)
    mask = scipy.ndimage.rotate(mask, 90, (1, 2))
    masks_dict[folder] = mask
    if i == 2:
      pass
    i += 1
  return masks_dict


def dicts_to_tensors(
    data_dict: Dict,
    masks_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Считанные словари данных коневертируются в тензорыа
  """
  data_tensor = torch.Tensor([])
  masks_tensor = torch.Tensor([])
  for key, val in data_dict.items():
    tmp_data = torch.Tensor(val)

    masks_dict[key][np.where(masks_dict[key] < 0.5)] = 0
    masks_dict[key][np.where(masks_dict[key] > 0.5)] = 1

    tmp_masks = torch.Tensor(masks_dict[key])
    data_tensor = torch.cat((data_tensor, tmp_data), dim=0)
    masks_tensor = torch.cat((masks_tensor, tmp_masks), dim=0)
  return data_tensor, masks_tensor


def plot_learning_step(val_data,
                       pred_data,
                       epoch,
                       num_epochs,
                       avg_loss,
                       test_loss):    
    clear_output(wait=True)
    plt.figure(figsize=(18, 6))
    for k in range(6):
        plt.subplot(2, 6, k + 1)
        plt.imshow(val_data[k, 0].numpy(), cmap='gray')
        plt.title('Real')
        plt.axis('off')

        plt.subplot(2, 6, k + 7)
        plt.imshow(pred_data[k, 0], cmap='gray')
        plt.title('Output')
        plt.axis('off')
    plt.suptitle('%d / %d - train loss: %f, test loss: %f' \
      % (epoch + 1, num_epochs, avg_loss, test_loss))
    plt.show()