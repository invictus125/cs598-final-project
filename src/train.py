import torch
from torch.optim import Adam
from torch.nn import BCELoss
import numpy as np
from torcheval.metrics import BinaryAUROC, BinaryAUPRC, BinaryRecall


def _binary_specificity(test, target):
  # TN / TN + FP
  pinned = torch.where(test >= 0.5, 1.0, 0.0)
  pos = torch.where(pinned > 0, 1.0, 0.0)
  neg = torch.where(pinned < 1, 1.0, 0.0)
  gt_pos = torch.where(target > 0, 1.0, 0.0)
  gt_neg = torch.where(target < 1, 1.0, 0.0)
  tn = neg + gt_neg
  tn = torch.sum(torch.where(tn > 1, 1.0, 0.0), dtype=torch.float)
  fp = pos + gt_neg
  fp = torch.sum(torch.where(fp > 1, 1.0, 0.0), dtype=torch.float)

  return (tn / (tn + fp))


def _train_one_epoch(
  model,
  train_data,
  optimizer,
  criterion
):
  model.train()
  loss_history = []
  for data in train_data:
    optimizer.zero_grad()
    y_hat = model(data.abp, data.ecg, data.eeg)
    loss = criterion(y_hat, data.y)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

  return loss_history


def _eval_model(
  model,
  eval_data,
  dataset_name
):
  model.eval()

  auroc = []
  auprc = []
  sensitivity = []
  specificity = []
  
  f_auroc = BinaryAUROC()
  f_auprc = BinaryAUPRC()
  f_sensitivity = BinaryRecall()
  for data in eval_data:
    y_hat = model(data.abp, data.ecg, data.eeg)
    
    f_auroc.update(y_hat, data.y)
    f_auprc.update(y_hat, data.y)
    f_sensitivity.update(y_hat, data.y)
    
    auroc.append(f_auroc.compute())
    auprc.append(f_auprc.compute())
    sensitivity.append(f_sensitivity.compute())
    specificity.append(_binary_specificity(y_hat, data.y))

  m_auroc = np.mean(auroc)
  m_auprc = np.mean(auprc)
  m_sensitivity = np.mean(sensitivity)
  m_specificity = np.mean(specificity)

  print(f'    {dataset_name} data metrics:')
  print(f'        AUROC: {m_auroc}')
  print(f'        AUPRC: {m_auprc}')
  print(f'        Sensitivity: {m_sensitivity}')
  print(f'        Specificity: {m_specificity}')

  return m_auroc, m_auprc, m_sensitivity, m_specificity


def train(
  model,
  train_data_handle,
  test_data_handle,
  learning_rate=0.0001,
  epochs=100,
  suspend_train_epochs_threshold=5
):
  """Trains an IntraoperativeHypotensionModel using the given learning rate for
  the given number of epochs

  model: the IntraoperativeHypotensionModel to train
  train_data_handle: the dataset we will train on
  test_data_handle: the dataset we will use for evaluation
  learning_rate: the learning rate to use with the Adam optimizer
  epochs: the number of epochs to train for
  suspend_train_epochs_threshold: training will be suspended if the loss does
    not improve for this number of epochs
  """
  if model is None or train_data_handle is None or test_data_handle is None:
    raise ValueError(
      'model, train_data_handle, and test_data_handle are required for training'
    )

  criterion = BCELoss()
  optimizer = Adam(model.parameters(), lr=learning_rate)

  overall_loss_history = []
  consecutive_epochs_without_improvement = 0
  for epoch in range(epochs):
    print('====================================')
    print(f'     Epoch #{epoch + 1}')
    print('====================================')
    loss_history = _train_one_epoch(
      model,
      train_data_handle,
      optimizer,
      criterion
    )
    _eval_model(model, train_data_handle, 'Train')
    # Not using performance metrics yet in this function.
    # Potential TODO: stop training once desired performance is reached (TBD)
    performance = _eval_model(model, test_data_handle, 'Test')
    
    if epoch > 0:
      mean_loss = np.mean(loss_history)
      overall_loss_history.append(mean_loss)
      loss_change = overall_loss_history[epoch - 1] - mean_loss
      if loss_change < 0.1:
        consecutive_epochs_without_improvement += 1
      else:
        consecutive_epochs_without_improvement = 0

    if consecutive_epochs_without_improvement >= suspend_train_epochs_threshold:
      print(f'Training stopping after {epoch+1} epochs.')
      print(f'Loss did not change for {suspend_train_epochs_threshold} epochs')
      break

