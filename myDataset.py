import torch

from torch.utils.data import Dataset
from torchvision import transforms


class TorchData(Dataset):
  """
  Датасет на основе Dataset библиотеки PyTorch.
  Добавлена возможность извлечения всех данных. Данные нормализуются внутри датасета.
  """

  def __init__(self, images, masks, transform=None):
    self.X = images.type(torch.FloatTensor)
    self.y = masks.type(torch.FloatTensor)
    self.transform = transforms.Normalize(0.485, 0.229) if not transform else transform
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.transform(self.X[index]), self.y[index]

  def __all_data__(self):
    return self.transform(self.X), self.y

  def __len__(self):
    return self.len

