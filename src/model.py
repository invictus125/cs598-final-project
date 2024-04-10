import torch
from torch import nn


# TODO: encoder layer


class ResidualBlock(nn.Module):
  def __init__(self, dim_in, kernel_size=15, stride=1):
    super(ResidualBlock, self).__init__()
    self.bn1 = nn.BatchNorm1d(dim_in)
    self.act1 = nn.ReLU()
    self.do = nn.Dropout()
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


class FlattenAndLinearBlock(nn.Module):
  def __init__(self, dim_in, dim_out):
    super(FlattenAndLinearBlock, self).__init__()
    self.fc = nn.Linear(dim_in, dim_out)

  
  def forward(self, x):
    x_hat = torch.flatten(x)
    x_hat = self.fc(x_hat)
    
    return x_hat


class WaveformResNet(nn.Module):
  def __init__(
    self,
    name,
    input_shape,
    output_size,
    num_residuals=12,
    filter_size=15,
    stride=1
  ):
    self.encoder = None # TODO
    self.res_in_dim = None # TODO - this should be the encoder's out dimension

    self.residuals = []
    for i in range(num_residuals):
      self.residuals.append(ResidualBlock((self.res_in_dim * (i+1)), filter_size, stride))

    self.fl_ln = FlattenAndLinearBlock((self.res_in_dim * num_residuals), output_size)

  
  def forward(self, x):
    x_hat = self.encoder(x)

    for i in range(len(self.residuals)):
      x_hat = self.residuals[i](x_hat)

    return self.fl_ln(x_hat)


# TODO: Create model with optional resnets for each input type and appropriate FC layers