import torch
from torch import nn

class ResidualBlock(nn.Module):
  def __init__(self, dim_in, kernel_size, stride):
    super(ResidualBlock, self).__init__()
    self.bn1 = nn.BatchNorm1d(dim_in)
    self.act1 = nn.ReLU()
    self.do = nn.Dropout()
    # TODO: verify what they did for convolutions
    self.conv1 = nn.Conv1d(dim_in, dim_in, kernel_size, stride)
    self.bn2 = nn.BatchNorm1d(dim_in)
    self.act2 = nn.ReLU()
    self.conv2 = nn.Conv1d(dim_in, dim_in, kernel_size, stride)


  def forward(self, x):
    x_hat = self.bn1(x)
    x_hat = self.act1(x_hat)
    x_hat = self.do(x_hat)
    x_hat = self.conv1(x_hat)
    x_hat = self.bn2(x_hat)
    x_hat = self.act2(x_hat)
    x_hat = self.conv2(x_hat)

    # Concat raw input with convolved for skip connection
    return torch.cat([x, x_hat], 1)


def create_1d_residual(dim_in, kernel_size, stride=1):
  print('hi')
  return ResidualBlock(dim_in, kernel_size, stride)