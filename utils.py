import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from scipy import io
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CustomDataSet(Dataset):
    # x_tensor: data
    # y_tensor: label
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def ManualSeed(seed: int, deterministic=False):
    # random seed 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:  # True면 cudnn seed 고정 (정확한 재현 필요한거 아니면 제외)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def doTrain(model: nn.Module,
            train_loader: DataLoader,
            num_epoch: int,
            optimizer: optim.Optimizer):
    # criterion = nn.BCELoss()
    # sigmoid = nn.Sigmoid()
    criterion = nn.CrossEntropyLoss()
    tr_acc = np.zeros(num_epoch)
    tr_loss = np.zeros(num_epoch)
    model.train()
    for epoch in range(num_epoch):
        correct, total, trn_loss, pred1 = (0, 0, 0.0, 0)
        for i, (x, y) in enumerate(train_loader, 0):
            x, y = (a.to(DEVICE) for a in [x, y])
            optimizer.zero_grad()
            out = model(x)
            # out = sigmoid(out)
            _, pred = torch.max(out.data, 1)
            # loss = criterion(out, y.float())
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            # pred = (out>0.5).int()
            pred1 += pred.sum().item()
            total += y.size(0)
            correct += (pred == y).sum().item()
            trn_loss += loss.item()

        tr_loss[epoch] = round(trn_loss / len(train_loader), 4)
        tr_acc[epoch] = round(100 * correct / total, 4)
        print(f'\n[{epoch:0>2}] acc: {tr_acc[epoch]} | loss: {tr_loss[epoch]} | predict 1, total: {pred1}, {total}')
    return tr_acc, tr_loss


def doTest(model: nn.Module, test_loader: DataLoader):
    # sigmoid = nn.Sigmoid()
    # sigmoid = nn.Softmax(dim=1)
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        model.eval()
        correct, total, pred1 = (0, 0, 0)
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            # predicted = (out > 0.5).int()
            _, predicted = torch.max(out.data, 1)
            pred1 += predicted.sum().item()
            # pred, _ = torch.max(sigmoid(out.data),1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            preds = np.append(preds, predicted.to('cpu').numpy())
            targets = np.append(targets, y.to('cpu').numpy())
    # preds = np.stack(preds,axis=0)
    # targets = np.stack(targets,axis=0)
    acc = round(100 * correct / total, 4)
    print(f'\n ***| FINAL ACC: {acc:.4f} |***\npredicted1, total: {pred1}, {total}')
    return acc, preds, targets


def SaveResults_mat(filepath, test_acc, test_preds, test_targets, tr_acc, tr_loss, num_batch, num_epoch, lr):
    path = './results/' + filepath + '.mat'
    io.savemat(path, {'acc': test_acc, 'preds': test_preds, 'targets': test_targets,
                      'tr_acc': tr_acc, 'tr_loss': tr_loss,
                      'info': f'batch_{num_batch}, epoch_{num_epoch}, lr_{lr}'})