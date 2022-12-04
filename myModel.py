import torch.nn.functional as F

import torch
from torch import nn


class Net(nn.Module):
  """
  Нейронная сеть. За базовую архитектуру была взята UNet.
  В данной реализации отсутствует на декодере батч-нормализация.
  Последний слой без функции активации.
  """
  def __init__(self):
    super(Net, self).__init__()
    #Encoder layers
    self.enc_conv_0_1 = nn.Conv2d(1, 32, 3, padding=1)
    self.enc_conv_0_2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                      nn.BatchNorm2d(32))
    self.enc_conv_1_1 = nn.Conv2d(32, 64, 3, padding=1)
    self.enc_conv_1_2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                      nn.BatchNorm2d(64))
    self.enc_conv_2_1 = nn.Conv2d(64, 128, 3, padding=1)
    self.enc_conv_2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                      nn.BatchNorm2d(128))

    self.maxpool = nn.MaxPool2d(2, return_indices=False)

    #bottleneck layer
    self.bottleneck_conv_0 = nn.Conv2d(128, 256, 3, padding=1)
    self.bottleneck_conv_1 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                           nn.BatchNorm2d(256))

    #Decoder layers
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    self.upconv = nn.Conv2d(256, 128, 3, padding=1)

    self.dec_conv_0_1 = nn.Conv2d(256, 128, 3, padding=1)
    self.dec_conv_0_2 = nn.Conv2d(128, 64, 3, padding=1)
    self.dec_conv_1_1 = nn.Conv2d(128, 64, 3, padding=1)
    self.dec_conv_1_2 = nn.Conv2d(64, 32, 3, padding=1)
    self.dec_conv_2 = nn.Conv2d(64, 1, 3, padding=1)

    self.bn1 = nn.BatchNorm2d(1)
    self.bn32 = nn.BatchNorm2d(32)
    self.bn64 = nn.BatchNorm2d(64)
    self.bn128 = nn.BatchNorm2d(128)
    self.bn256 = nn.BatchNorm2d(256)
    self.bn512 = nn.BatchNorm2d(512)

  def forward(self, x):
    #Data encoding 
    e0 = F.relu(self.enc_conv_0_1(x))
    e0 = F.relu(self.enc_conv_0_2(e0))

    e1 = F.relu(self.enc_conv_1_1(self.maxpool(e0)))
    e1 = F.relu(self.enc_conv_1_2(e1))

    e2 = F.relu(self.enc_conv_2_1(self.maxpool(e1)))
    e2 = F.relu(self.enc_conv_2_2(e2))

    #bottleneck
    b0 = F.relu(self.bottleneck_conv_0(self.maxpool(e2)))
    b1 = F.relu(self.bottleneck_conv_1(b0))
  
    
    #data decoding & concatination
    u0 = self.upconv(self.upsample(b1))
    crop_idxs = abs(e2.size(2) - u0.size(2)) // 2
    e2 = e2[:, :, crop_idxs:crop_idxs + u0.size(2), crop_idxs:crop_idxs + u0.size(2)]
    d0_cat = torch.cat((e2, u0), dim=1)
    d0 = F.relu(self.dec_conv_0_1(d0_cat))
    d0 = F.relu(self.dec_conv_0_2(d0))
    
    u1 = self.upsample(d0)
    crop_idxs = abs(e1.size(2) - u1.size(2)) // 2
    e1 = e1[:, :, crop_idxs:crop_idxs + u1.size(2), crop_idxs:crop_idxs + u1.size(2)]
    d1_cat = torch.cat((e1, u1), dim=1)
    d1 = F.relu(self.dec_conv_1_1(d1_cat))
    d1 = F.relu(self.dec_conv_1_2(d1))

    u2 = self.upsample(d1)
    crop_idxs = abs(e0.size(2) - u2.size(2)) // 2
    e0 = e0[:, :, crop_idxs:crop_idxs + u2.size(2), crop_idxs:crop_idxs + u2.size(2)]
    d2_cat = torch.cat((e0, u2), dim=1)
    d2 = self.dec_conv_2(d2_cat)

    return d2