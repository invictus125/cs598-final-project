import torch
from torch import nn


class EncoderBlock(nn.Module):
  def __init__(self, dim_in, kernel_size=15, stride=1):
    super(EncoderBlock, self).__init__()
    self.conv = nn.Conv1d(1, 1, kernel_size, stride, padding=7)
    self.fc = nn.Linear(dim_in, dim_in)


  def forward(self, x):
    x_hat = torch.flatten(self.conv(x))
    return self.fc(x_hat)


class ResidualBlock(nn.Module):
  def __init__(
    self,
    in_channels,
    out_channels,
    size_down,
    kernel_size,
    stride=1
  ):
    super(ResidualBlock, self).__init__()
    self.bn1 = nn.BatchNorm1d(in_channels)
    self.act1 = nn.ReLU()
    self.do = nn.Dropout()
    self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride)
    self.bn2 = nn.BatchNorm1d(in_channels)
    self.act2 = nn.ReLU()
    self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)

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
    data_type
  ):
    super(WaveformResNet, self).__init__()
    self.encoder = EncoderBlock(input_shape, 15, 1)
    self.res_in_dim = input_shape
    self.output_size = output_size

    if data_type not in ['abp', 'ecg', 'eeg']:
      raise ValueError('Invalid data type. Must be one of [abp, ecg, eeg]')

    # Set up configurations for residual blocks
    residual_configs = []
    linear_block_input_length = -1
    if data_type in ['abp', 'ecg']:
      residual_configs = [
        {
          'kernel_size': 15,
          'in_channels': 1,
          'out_channels': 2,
          'size_down': True,
        },
        {
          'kernel_size': 15,
          'in_channels': 2,
          'out_channels': 2,
          'size_down': False,
        },
        {
          'kernel_size': 15,
          'in_channels': 2,
          'out_channels': 2,
          'size_down': True,
        },
        {
          'kernel_size': 15,
          'in_channels': 2,
          'out_channels': 2,
          'size_down': False,
        },
        {
          'kernel_size': 15,
          'in_channels': 2,
          'out_channels': 2,
          'size_down': True,
        },
        {
          'kernel_size': 15,
          'in_channels': 2,
          'out_channels': 4,
          'size_down': False,
        },
        {
          'kernel_size': 7,
          'in_channels': 4,
          'out_channels': 4,
          'size_down': True,
        },
        {
          'kernel_size': 7,
          'in_channels': 4,
          'out_channels': 4,
          'size_down': False,
        },
        {
          'kernel_size': 7,
          'in_channels': 4,
          'out_channels': 4,
          'size_down': True,
        },
        {
          'kernel_size': 7,
          'in_channels': 4,
          'out_channels': 6,
          'size_down': False,
        },
        {
          'kernel_size': 7,
          'in_channels': 4,
          'out_channels': 6,
          'size_down': True,
        },
        {
          'kernel_size': 7,
          'in_channels': 6,
          'out_channels': 6,
          'size_down': False,
        },
      ]
      linear_block_input_length = 496
    else:
      residual_configs = [
        {
          'kernel_size': 7,
          'in_channels': 1,
          'out_channels': 2,
          'size_down': True,
        },
        {
          'kernel_size': 7,
          'in_channels': 2,
          'out_channels': 2,
          'size_down': False,
        },
        {
          'kernel_size': 7,
          'in_channels': 2,
          'out_channels': 2,
          'size_down': True,
        },
        {
          'kernel_size': 7,
          'in_channels': 2,
          'out_channels': 2,
          'size_down': False,
        },
        {
          'kernel_size': 7,
          'in_channels': 2,
          'out_channels': 2,
          'size_down': True,
        },
        {
          'kernel_size': 7,
          'in_channels': 2,
          'out_channels': 4,
          'size_down': False,
        },
        {
          'kernel_size': 3,
          'in_channels': 4,
          'out_channels': 4,
          'size_down': True,
        },
        {
          'kernel_size': 3,
          'in_channels': 4,
          'out_channels': 4,
          'size_down': False,
        },
        {
          'kernel_size': 3,
          'in_channels': 4,
          'out_channels': 4,
          'size_down': True,
        },
        {
          'kernel_size': 3,
          'in_channels': 4,
          'out_channels': 6,
          'size_down': False,
        },
        {
          'kernel_size': 3,
          'in_channels': 6,
          'out_channels': 6,
          'size_down': True,
        },
        {
          'kernel_size': 3,
          'in_channels': 6,
          'out_channels': 6,
          'size_down': False,
        },
      ]
      linear_block_input_length = 120

    self.residuals = []
    # Build residuals
    for i in range(12):
      self.residuals.append(
        ResidualBlock(
          size_down=residual_configs[i]['size_down'],
          in_channels=residual_configs[i]['in_channels'],
          out_channels=residual_configs[i]['out_channels'],
          kernel_size=residual_configs[i]['kernel_size'],
        )
      )

    self.fl_ln = FlattenAndLinearBlock(linear_block_input_length, output_size)

  
  def forward(self, x):
    x_hat = self.encoder(x).unsqueeze(0)

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
