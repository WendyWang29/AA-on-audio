import sys

import numpy as np
import pandas as pd
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

from src.resnet_features import get_log_spectrum, compute_mfcc_feats
from src.resnet_model import SpectrogramModel, MFCCModel
from src.utils import *
from src.audio_utils import read_audio


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


def evaluate_accuracy_resnet(data_loader, model, device):
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


def evaluate_metrics(data_loader, model, device):
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
        batch_out = model(batch_x)
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


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    predictions = pd.DataFrame(columns=['Path', 'Bin_label', 'Prediction', 'Pred_0', 'Pred_1'])
    for tracks, batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1] - batch_out[:, 0]
                       ).data.cpu().numpy().ravel()

        pred_dict = {"Path": np.array(list(tracks)), "Bin_label":batch_y.numpy().ravel(), "Prediction": batch_score,
                     'Pred_0':(batch_out[:, 1]).data.cpu().numpy().ravel(), 'Pred_1':(batch_out[:, 0]).data.cpu().numpy().ravel()}
        predictions = predictions.append(pd.DataFrame(pred_dict))

    predictions.to_csv(save_path)
    print('Result saved to {}'.format(save_path))


def train_epoch_resnet(data_loader, model, lr, device):
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
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out .max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy


def get_loss_resnet(data_loader, model, device):
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
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out .max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
    running_loss /= num_total
    return running_loss


def get_features(wav_path, features, args, X, cached=False, force=False):
    """
    Extract features chosen by features argument.

    :param wav_path: filename of the audio
    :type wav_path: str
    :param features: name of the features to be computed in [lfcc]
    :type features: str
    :param args: configuration dictionary
    :type args: dict
    :return: extracted features
    :rtype np.array
    """
    def get_feats(wav_path, X):

        # if X.all() == None:
        X, fs = read_audio(wav_path)
        args['win_len'] = None
        args['hop_size'] = None

        if features == 'spec':
            return get_log_spectrum(wav_path, X, args['fsamp'], win_len=args['win_len'], hop_size=args['hop_size'])
        elif features == 'mfcc':
            return compute_mfcc_feats(wav_path, X, args['fsamp'], win_len=args['win_len'], hop_size=args['hop_size'])
        else:
            raise ValueError('Feature type not supported.')

    if cached:
        cache_dir = args['cache_dir'] + features
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        file_name = os.path.join(cache_dir, os.path.splitext(os.path.basename(wav_path))[0] + '.npy')
        if not os.path.exists(file_name):# or force:
            data = get_feats(wav_path, X)
            np.save(file_name, data)
            return data
        else:
            try:
                data = np.load(file_name, allow_pickle=True)
            except:
                data = get_feats(wav_path, X)
                np.save(file_name, data)
            return data

    else:
        return get_feats(wav_path, X)


class LoadTrainData_ResNet(Dataset):
    def __init__(self, list_IDs, labels, win_len, config):
        """
        self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)
        """

        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.config = config


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]
        y = self.labels[track]
        X = get_features(track, self.config['features'], self.config, X=None, cached=True)

        feature_len = X.shape[1]
        
        network_input_shape = 28 * self.win_len

        if feature_len < network_input_shape:
            num_repeats = int(network_input_shape/feature_len) + 1
            X = np.tile(X, (1, num_repeats))
            # feature_len = X.shape[1]

        X_win = X[:, : network_input_shape]
        X_win = Tensor(X_win)

        # last_valid_start_sample = feature_len - network_input_shape
        # if not last_valid_start_sample == 0:
        #     start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        # else:
        #     start_sample = 0
        # X_win = X[:, start_sample: start_sample + network_input_shape]
        # X_win = Tensor(X_win)

        return X_win, y


class LoadEvalData_ResNet(Dataset):
    def __init__(self, list_IDs, win_len, config):
        """
        self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)
        """

        self.list_IDs = list_IDs
        self.win_len = win_len
        self.config = config

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]
        X = get_features(track, self.config['features'], self.config, X=None, cached=True)

        feature_len = X.shape[1]

        network_input_shape = 28 * self.win_len

        if feature_len < network_input_shape:
            num_repeats = int(network_input_shape/feature_len) + 1
            X = np.tile(X, (1, num_repeats))

        X_win = X[:, : network_input_shape]
        X_win = Tensor(X_win)

        return X_win, track


def eval():

    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_everything(1234)

    config_path = '../config/residualnet_train_config.yaml'
    config_res = read_yaml(config_path)

    assert config_res['features'] in ['mfcc', 'spec'], 'Not supported feature'

    if config_res['features'] == 'mfcc':
        model_cls = MFCCModel
    elif config_res['features'] == 'spec':
        model_cls = SpectrogramModel

    model = model_cls().to(device)

    model_path = os.path.join(config_res['model_folder'], config_res['model_path'])
    model.load_state_dict(torch.load(model_path))
    print('Model loaded : {}'.format(model_path))

    df_eval = pd.read_csv(config_res['df_eval_path'], index_col=0)

    d_label_eval = dict(zip(df_eval['Path'], df_eval['Bin_label']))
    file_eval = list(df_eval['Path'])
    eval_set = LoadEvalData(list_IDs=file_eval, labels=d_label_eval, win_len=config_res['win_len'], config=config_res)
    eval_loader = DataLoader(eval_set, batch_size=config_res['batch_size'], shuffle=True, num_workers=32)
    # del eval_set, d_label_eval

    eval_path = os.path.join(config_res['eval_output'], 'prediction_{}.csv'.format(config_res['features']))
    produce_evaluation_file(eval_set, model, device, eval_path)
