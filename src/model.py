import torch
from torch import nn


class EncoderBlock(nn.Module):
  def __init__(self, dim_in, kernel_size=15, stride=1):
    super(EncoderBlock, self).__init__()
    self.conv = nn.Conv1d(dim_in, dim_in, kernel_size, stride)
    self.fc = nn.Linear(dim_in, dim_in)


  def forward(self, x):
    return self.fc(self.conv(x))


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
    self.encoder = EncoderBlock(input_shape, 15, 1)
    self.res_in_dim = input_shape
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

  
  def get_output_size(self):
    return self.output_size


class IntraoperativeHypotensionModel(nn.Module):
  def __init__(
    self,
    ecg_resnet,
    abp_resnet,
    eeg_resnet
  ):
    super(IntraoperativeHypotensionModel, self).__init__()

    self.ecg = ecg_resnet
    self.abp = abp_resnet
    self.eeg = eeg_resnet

    self.fc_input_length = 0

    if self.ecg is not None:
      self.fc_input_length += self.ecg.get_output_size()

    if self.abp is not None:
      self.fc_input_length += self.abp.get_output_size()

    if self.eeg is not None:
      self.fc_input_length += self.eeg.get_output_size()

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

  
  def forward(self, abp, ecg, eeg):
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
