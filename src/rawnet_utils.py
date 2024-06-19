import sys

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
import librosa
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def get_waveform(wav_path, config):
    fs = 16000
    X, fs_orig = librosa.load(wav_path, sr=None, duration=240, mono=True)
    if fs_orig != fs:
        X = librosa.resample(X, fs_orig, fs)
    return X

def create_mini_batch_RawNet(audio):
    feature_len = audio.shape[0]
    network_input_shape = 16000 * 4
    if feature_len < network_input_shape:
        num_repeats = int(network_input_shape / feature_len) + 1
        audio = np.tile(audio, num_repeats)
    X_win = audio[: network_input_shape]
    X_win = np.expand_dims(X_win, axis=0)
    X_win = Tensor(X_win)
    return X_win # the mini batch, still on CPU

def train_epoch_rawnet(data_loader, model, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()
    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x) #removed batch_y
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


def evaluate_accuracy_rawnet(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
    # for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

    return 100 * (num_correct / num_total)



class LoadEvalData_RawNet(Dataset):
    def __init__(self, list_IDs, config):
        """
        self.list_IDs	: list of strings (each string: utt key),
        """

        self.list_IDs = list_IDs
        self.win_len = 4
        self.config = config


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]
        X = get_waveform(wav_path=track, config=self.config)
        feature_len = X.shape[0]
        network_input_shape = 16000 * self.win_len
        if feature_len < network_input_shape:
            num_repeats = int(network_input_shape/feature_len) + 1
            X = np.tile(X, num_repeats)
            # feature_len = X.shape[1]

        X_win = X[: network_input_shape]
        X_win = Tensor(X_win)

        return X_win, track


class LoadTrainData_RawNet(Dataset):
    def __init__(self, list_IDs, labels, config):
        """
        self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)
        """

        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = 4
        self.config = config


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]
        y = self.labels[track]
        X = get_waveform(wav_path=track, config=self.config)
        feature_len = X.shape[0]
        network_input_shape = 16000 * self.win_len
        if feature_len < network_input_shape:
            num_repeats = int(network_input_shape/feature_len) + 1
            X = np.tile(X, num_repeats)
            # feature_len = X.shape[1]

        X_win = X[: network_input_shape]
        X_win = Tensor(X_win)

        #assert X_win.shape == (64000,), f'Expected shape (64000,) but got {X_win.shape}'

        return X_win, y



class LoadAttackData_RawNet(Dataset):
    def __init__(self, list_IDs, labels, config):
        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = 4
        self.config = config

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]
        y = self.labels[track]
        X = get_waveform(wav_path=track, config=self.config)
        feature_len = X.shape[0]
        network_input_shape = 16000 * self.win_len
        if feature_len < network_input_shape:
            num_repeats = int(network_input_shape / feature_len) + 1
            X = np.tile(X, num_repeats)
            # feature_len = X.shape[1]

        X_win = X[: network_input_shape]
        X_win = Tensor(X_win)

        return X_win, y, feature_len, index

