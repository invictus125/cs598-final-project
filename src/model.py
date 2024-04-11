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

    self.seq = nn.Sequential(
      self.bn1,
      self.act1,
      self.do,
      self.conv1,
      self.bn2,
      self.act2,
      self.conv2
    )


  def forward(self, x):
    # x_hat = self.bn1(x)
    # x_hat = self.act1(x_hat)
    # x_hat = self.do(x_hat)
    # x_hat = self.conv1(x_hat)
    # x_hat = self.bn2(x_hat)
    # x_hat = self.act2(x_hat)
    # x_hat = self.conv2(x_hat)
    x_hat = self.seq(x)

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
    input_shape,
    output_size,
    num_residuals=12,
    filter_size=15,
    stride=1
  ):
    super(WaveformResNet, self).__init__()
    self.encoder = None # TODO
    self.res_in_dim = 1 # TODO - this should be the encoder's out dimension
    self.output_size = output_size

    self.residuals = []
    for i in range(num_residuals):
      self.residuals.append(ResidualBlock((self.res_in_dim * (i+1)), filter_size, stride))

    self.fl_ln = FlattenAndLinearBlock((self.res_in_dim * num_residuals), output_size)

  
  def forward(self, x):
    x_hat = self.encoder(x)

    for i in range(len(self.residuals)):
      x_hat = self.residuals[i](x_hat)

    return self.fl_ln(x_hat)

  
  def output_size(self):
    return self.output_size


class IntraoperativeHypotensionModel(nn.Module):
  def __init__(
    self,
    ecgResNet,
    abpResNet,
    eegResNet
  ):
    super(IntraoperativeHypotensionModel, self).__init__()

    self.ecg = ecgResNet
    self.abp = abpResNet
    self.eeg = eegResNet

    self.fc_input_length = 0

    if self.ecg is not None:
      self.fc_input_length += self.ecg.output_size()

    if self.abp is not None:
      self.fc_input_length += self.abp.output_size()

    if self.eeg is not None:
      self.fc_input_length += self.eeg.output_size()

    if self.fc_input_length == 0:
      raise 'No resnet blocks provided, unable to build model'

    self.fc1 = nn.Linear(self.fc_input_length, 16)
    self.fc2 = nn.Linear(16, 1)
    self.act = nn.Sigmoid()

    self.seq = nn.Sequential(
      self.fc1,
      self.fc2,
      self.act
    )

  
  def forward(self, ecg, abp, eeg):
    ecg_o = torch.Tensor([])
    abp_o = torch.Tensor([])
    eeg_o = torch.Tensor([])
  
    if self.ecg is not None:
      ecg_o = self.ecg(ecg)

    if self.abp is not None:
      abp_o = self.abp(abp)

    if self.eeg is not None:
      eeg_o = self.eeg(eeg)

    fc_in = torch.flatten(torch.concat([ecg_o, abp_o, eeg_o]))

    prediction = self.seq(fc_in)

    return prediction
