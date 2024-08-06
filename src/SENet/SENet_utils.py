import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def train_epoch_SENet(data_loader, model, lr, device):
    '''
        it's the same as for ResNet except that we are passing an
        unsqueezed version of the batch to the model
    '''
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.NLLLoss(weight)
    # criterion = nn.NLLLoss()

    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
        # for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x.unsqueeze(dim=1))
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct / num_total) * 100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy

def evaluate_accuracy_SENet(data_loader, model, device):
    '''
        it's the same as for ResNet except that we are passing an
        unsqueezed version of the batch to the model
    '''
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
    # for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x.unsqueeze(dim=1))
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

    return 100 * (num_correct / num_total)

def get_loss_SENet(data_loader, model, device):
    '''
        it's the same as for ResNet except that we are passing an
        unsqueezed version of the batch to the model
    '''
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.NLLLoss(weight)
    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x.unsqueeze(dim=1))
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out .max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
    running_loss /= num_total
    return running_loss

def evaluate_metrics_SENet(data_loader, model, device):
    '''
        it's the same as for ResNet except that we are passing an
        unsqueezed version of the batch to the model
    '''
    model.eval()
    num_correct = 0.0
    num_total = 0.0
    roc_auc = np.zeros((data_loader.dataset.__len__(),))
    eer = np.zeros((data_loader.dataset.__len__()),)
    batch_index = 0
    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
    # for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x.unsqueeze(dim=1))
        _, batch_pred = batch_out.max(dim=1)
        if torch.unique(batch_y).size(dim=0) > 1:
            fpr, tpr, _ = roc_curve(batch_y.cpu().detach().numpy(),
                                    torch.exp(batch_out)[:, 1].cpu().detach().numpy())
            roc_auc[batch_index] = roc_auc_score(batch_y.cpu().detach().numpy(),
                                    torch.exp(batch_out)[:, 1].cpu().detach().numpy())
            fnr = 1 - tpr
            eer[batch_index] = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        else:
            roc_auc[batch_index] = np.nan
            eer[batch_index] = np.nan

        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        num_total += batch_size
        batch_index += 1
    accuracy = 100 * (num_correct / num_total)

    return np.nanmean(roc_auc), np.nanmean(eer), accuracy

