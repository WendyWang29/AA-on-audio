"""

"""
import torch
import os
import pandas as pd
import numpy as np
import torch.nn as nn
from src.resnet_model import SpectrogramModel
from src.resnet_utils import get_features


def load_spec_model(device, config):
    # TODO delete from tests or something
    """
    Load the spectrogram model - pre-trained
    :param device: GPU or CPU
    :param config: config file path
    :return: model
    """
    resnet_spec_model = SpectrogramModel().to(device)
    resnet_spec_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device))
    return resnet_spec_model

def retrieve_single_cached_spec(config, index):
    df_eval = pd.read_csv(os.path.join('..', config["df_eval_path"]))

    # get list of all eval files and labels
    file_eval = list(df_eval['path'])
    label_eval = list(df_eval['label'])

    # get one single file and its label given an index
    file = file_eval[index]
    label = label_eval[index]
    print(f'Evaluating file {file} with label {label}')

    # retrieve the spectrogram
    spec = get_features(wav_path=file,
                        features=config['features'],
                        args=config,
                        X=None,
                        cached=True,
                        force=False)

    return file, label, spec

def get_mini_batch(spec, device):
    """
    Given one single spec we create a mini batch that can be passed to the model
    :param spec: single spectrogram
    :param device:
    :return: tensor of size ([1, freq_bins, time_frames)
    """
    mini_batch = np.expand_dims(spec, axis=0)  # ndarray
    mini_batch = torch.from_numpy(mini_batch).to(device) # tensor
    print(f'The mini batch is {mini_batch.size()}')

    return mini_batch

def get_pred_class(pred):
    score = pred[0,0] - pred[0,1]
    if score > 0:
        pred = 0
    else:
        pred = 1

    return pred

def FGSM_perturb(spec, model, epsilon, GT, device):
    spec.requires_grad = True
    out = model(spec)
    pred = get_pred_class(out)
    print(f'The clean file is predicted to be class {pred}. The GT class is {GT}')

    L = nn.NLLLoss()
    loss = None
    label = torch.tensor([GT]).to(device)
    loss = L(out, label)
    model.zero_grad()

    loss.backward()
    grad = spec.grad

    # apply the perturbation
    p_spec = spec + epsilon * grad.sign()
    p_spec = p_spec.squeeze(0).detach
    p_spec = p_spec.cpu().numpy()

    return p_spec



class FGSM_attack():
    def __init__(self, config, model, device, epsilon):
        self.config = config
        self.model = model
        self.device = device
        self.epsilon = epsilon

    def attack_single_cached_spec(self, index):
        file, label, spec = retrieve_single_cached_spec(index, self.config)
        spec = get_mini_batch(spec, self.device)
        perturbed_spec = FGSM_perturb(spec, self.model, self.epsilon, label, self.device)
        # TODO find way to save the perturbed spec in folder









