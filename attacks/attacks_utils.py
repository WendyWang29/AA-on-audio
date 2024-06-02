"""
ATTACK SCRIPTS
"""

import os
import pandas as pd
import numpy as np
import torch
import sys
import librosa
import logging
import soundfile as sf
import torch.nn as nn
from tqdm import tqdm
from src.resnet_model import SpectrogramModel
from torch.utils.data import DataLoader
from src.resnet_utils import LoadAttackData_ResNet
from src.rawnet_utils import LoadAttackData_RawNet
from src.resnet_utils import get_features
from src.audio_utils import read_audio
from attacks.sp_utils import spectrogram_inversion_batch, spectrogram_inversion, get_spectrogram_from_audio
from src.resnet_features import compute_spectrum

logging.getLogger('numba').setLevel(logging.WARNING)



def load_spec_model(device, config):
    # TODO delete from tests or something
    """
    Load the spectrogram model - pre-trained
    :param device: GPU or CPU
    :param config: config file path
    :return: model
    """
    resnet_spec_model = SpectrogramModel().to(device)
    resnet_spec_model.load_state_dict(torch.load(os.path.join('..', config["model_path_spec"]), map_location=device))
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

def spec_to_tensor(spec, device):
    X_p_batch = np.expand_dims(spec, axis=0)  # ndarray
    X_p_batch_tensor = torch.from_numpy(X_p_batch).to(device)  # tensor
    X_p = X_p_batch_tensor
    X_p.requires_grad = True
    return X_p



def save_perturbed_audio(file, folder, audio, sr, epsilon, attack):



    # ensure folder path exists
    os.makedirs(folder, exist_ok=True)

    # create the file path
    file_name = os.path.splitext(os.path.basename(file))[0]
    file_path = os.path.join(folder, f'{attack}_{file_name}_{epsilon_str}.flac')

    # check if the same file already exists. If yes remove the old one
    if os.path.exists(file_path):
        os.remove(file_path)
        #print(f'Removed existing file: {file_path}')

    sf.write(file_path, audio, sr, format='FLAC')
    #print(f'Saved the perturbed audio as: {file_path}')

def save_perturbed_spec(file, folder, spec, epsilon, attack):
    '''
    Save the perturbed spec in the appropriate folder
    :param file: the file path to the flac file (to extract file name)
    :param folder: path to the folder in which file is to be saved
    :param spec: perturbed spectrogram to be saved as .npy file
    :param epsilon: epsilon used for the attack
    :param attack: name of the attack
    :return: None
    '''

    epsilon_str = str(epsilon).replace('.', 'dot')

    # ensure folder path exists
    os.makedirs(folder, exist_ok=True)

    # create the file path
    file_name = os.path.splitext(os.path.basename(file))[0]
    file_path = os.path.join(folder, f'{attack}_{file_name}_{epsilon_str}.npy')

    # check if the same file already exists. If yes remove the old one
    if os.path.exists(file_path):
        os.remove(file_path)
        #print(f'Removed existing file: {file_path}')

    np.save(file_path, spec)
    #print(f'Saved the perturbed spec as: {file_path}')

def evaluate_spec(spec, model, device):
    spec = spec_to_tensor(spec,device)
    out = model(spec)
    return out




def FGSM_perturb(spec, model, epsilon, GT, device):
    # Based on https://github.com/ymerkli/fgsm-attack/blob/master/fgsm_attack.py

    print('Performing the classic FGSM perturbation')

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
    grad = spec.grad.data

    # apply the perturbation
    p_spec = spec + epsilon * grad.sign()
    p_spec = p_spec.squeeze(0).detach()
    p_spec = p_spec.cpu()
    p_spec = p_spec.numpy()

    phase = None

    return p_spec, phase


#def FGSM_perturb_batch(data_loader, model, epsilon, config, device, folder_audio, folder_spec):
def FGSM_perturb_batch(data_loader, model, epsilon, config, device, folder_audio):
    print('FGSM attack starts...')

    df_eval = pd.read_csv(os.path.join('..', config["df_eval_path"]))
    file_eval = list(df_eval['path'])

    L = nn.NLLLoss()

    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True
        out = model(batch_x)
        loss = L(out, batch_y)
        model.zero_grad()
        loss.backward()
        grad = batch_x.grad.data
        perturbed_batch = batch_x + epsilon * grad.sign()

        perturbed_batch = perturbed_batch.squeeze(0).detach()
        perturbed_batch = perturbed_batch.cpu()
        perturbed_batch = perturbed_batch.numpy()

        for i in range(perturbed_batch.shape[0]):
            # working on each row of the matrix of perturbed specs
            sliced_spec = perturbed_batch[i][:, :time_frames[i]]
            # save_perturbed_spec(file=file_eval[index[i]],
            #                     folder=folder_spec,
            #                     spec=sliced_spec,
            #                     epsilon=epsilon,
            #                     attack='FGSM')

            audio, _ = spectrogram_inversion_batch(config=config,
                                             index=index[i],
                                             spec=sliced_spec,
                                             phase_info=True)

            save_perturbed_audio(file=file_eval[index[i]],
                                 folder=folder_audio,
                                 audio=audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='FGSM')


def FGSM_perturb_batch_RawNet(data_loader, model, epsilon, config, device, folder_audio):
    print('FGSM attack using RawNet2 starts...')

    df_eval = pd.read_csv(os.path.join('..', config["df_eval_path"]))
    file_eval = list(df_eval['path'])
    torch.backends.cudnn.enabled = False

    L = nn.NLLLoss()

    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):
        #TODO time frames should be called 'time samples'
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True
        out = model(batch_x)
        loss = L(out, batch_y)
        model.zero_grad()
        loss.backward()
        grad = batch_x.grad.data
        perturbed_batch = batch_x + epsilon * grad.sign()

        perturbed_batch = perturbed_batch.squeeze(0).detach()
        perturbed_batch = perturbed_batch.cpu()
        perturbed_batch = perturbed_batch.numpy()

        for i in range(perturbed_batch.shape[0]):
            # working on each row of the matrix of perturbed audios
            sliced_audio = perturbed_batch[i][:time_frames[i]]
            save_perturbed_audio(file=file_eval[index[i]],
                                folder=folder_audio,
                                audio=sliced_audio,
                                sr=16000,
                                epsilon=epsilon,
                                attack='FGSM_RawNet')









class Attack:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device

    def attack_single(self, index):
        # given the index retrieve the file path, the GT label and the spectrogram
        file, label, spec = retrieve_single_cached_spec(index=index,
                                                        config=self.config)

        ########
        feature_len = spec.shape[1]
        network_input_shape = 28 * 3
        if feature_len < network_input_shape:
            num_repeats = int(network_input_shape / feature_len) + 1
            spec = np.tile(spec, (1, num_repeats))
        spec = spec[:, :network_input_shape]
        ########


        # turn the single spec into a mini-batch to be fed to the model
        spec = get_mini_batch(spec, self.device)
        return spec, label, file

    def evaluate_single(self, label, perturbed_spec, audio_path, perturbed_audio):
        # the perturbed spectrogram is directly classified (no audio conversion)
        out_spec = evaluate_spec(perturbed_spec, self.model, self.device)
        print(f'The model output for the perturbed spectrogram is: {out_spec}')
        print(f'The perturbed spectrogram is predicted as: {get_pred_class(out_spec)}. GT label is {label}')

        # from the perturbed audio we compute again the spectrogram
        #perturbed_audio_spec = get_spectrogram_from_audio(audio_path)
        spec = compute_spectrum(perturbed_audio)
        out_audio = evaluate_spec(spec, self.model, self.device)
        print(f'The model output for the perturbed audio is: {out_audio}')
        print(f'The perturbed audio is predicted as: {get_pred_class(out_audio)}. GT label is {label}')

    def attack_dataset(self, eval_csv, model='RawNet'):
        eval_labels = dict(zip(eval_csv['path'], eval_csv['label']))
        file_eval = list(eval_csv['path'])

        # get the data loader and return the dataloader
        if model == 'ResNet':
            feat_set = LoadAttackData_ResNet(list_IDs=file_eval,
                                             labels=eval_labels,
                                             win_len=self.config['win_len'],
                                             config=self.config)
            feat_loader = DataLoader(feat_set,
                                     batch_size=self.config['eval_batch_size'],
                                     shuffle=False,
                                     num_workers=15)
            del feat_set, eval_labels
        elif model == 'RawNet':
            feat_set = LoadAttackData_RawNet(list_IDs=file_eval,
                                             labels=eval_labels,
                                             config=self.config)
            feat_loader = DataLoader(feat_set,
                                     batch_size=self.config['eval_batch_size'],
                                     shuffle=False,
                                     num_workers=15)
            del feat_set, eval_labels

        return feat_loader


class FGSMAttack(Attack):
    def __init__(self, epsilon, config, model, device):
        super().__init__(config, model, device)
        self.epsilon = epsilon

    def attack_single(self, index, type_of_attack):
        spec, label, file = super().attack_single(index)

        if type_of_attack == 'FGSM':
            spec_folder = 'p_specs'
            audio_folder = 'p_audio'
        else:
            print('Invalid type of attack')
            sys.exit(1)

        # define the paths for saving the perturbed data
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.folder_specs_FGSM = os.path.join(self.current_dir, 'FGSM_data', spec_folder)
        self.folder_audio_FGSM = os.path.join(self.current_dir, 'FGSM_data', audio_folder)

        """
        'FGSM' = run the classic FGSM attack
        """
        if type_of_attack == 'FGSM':
            perturbed_spec, phase = FGSM_perturb(spec, self.model, self.epsilon, label, self.device)
        else:
            print('No attack')


        # retrieve the perturbed audio
        perturbed_audio, _ = spectrogram_inversion(self.config,
                                                index,
                                                perturbed_spec,
                                                phase_info=True,
                                                phase_to_use=phase)

        # save the files
        save_perturbed_spec(file=file,
                            folder=self.folder_specs_FGSM,
                            spec=perturbed_spec,
                            epsilon=self.epsilon,
                            attack=type_of_attack)

        save_perturbed_audio(file=file,
                             folder=self.folder_audio_FGSM,
                             audio=perturbed_audio,
                             sr=16000,
                             epsilon=self.epsilon,
                             attack=type_of_attack)

        self.evaluate_single(label, perturbed_spec, file, perturbed_audio)

    def attack_dataset(self, eval_csv):
        feat_loader = super().attack_dataset(eval_csv)

        # create folder in which I save the perturbed audios (if it does not already exist)
        epsilon = str(self.epsilon).replace('.', 'dot')
        audio_folder = f'FGSM_dataset_{epsilon}'
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio_folder = os.path.join(self.current_dir, 'FGSM_data', audio_folder)
        os.makedirs(self.audio_folder, exist_ok=True)
        print(f'Saving the perturbed dataset in {self.audio_folder}')

        # create folder in which I save the perturbed specs (if it does not already exist)
        # epsilon = str(self.epsilon).replace('.', 'dot')
        # spec_folder = f'FGSM_dataset_{epsilon}_specs'
        # self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # self.spec_folder = os.path.join(self.current_dir, 'FGSM_data', spec_folder)
        # os.makedirs(self.spec_folder, exist_ok=True)
        # print(f'Saving the perturbed dataset specs in {self.spec_folder}')

        # perform the attack on batches (given by the data loader)
        FGSM_perturb_batch(feat_loader, self.model, self.epsilon, self.config, self.device, self.audio_folder)
        # FGSM_perturb_batch(feat_loader, self.model, self.epsilon, self.config, self.device, self.audio_folder, self.spec_folder)

    def attack_dataset_RawNet(self, eval_csv):
        feat_loader = super().attack_dataset(eval_csv)

        # create folder in which I save the perturbed audios (if it does not already exist)
        epsilon = str(self.epsilon).replace('.', 'dot')
        audio_folder = f'FGSM_RawNet_dataset_{epsilon}'
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio_folder = os.path.join(self.current_dir, 'FGSM_data', audio_folder)
        os.makedirs(self.audio_folder, exist_ok=True)
        print(f'Saving the perturbed dataset in {self.audio_folder}')

        # perform the attack on batches (given by the data loader)
        FGSM_perturb_batch_RawNet(feat_loader, self.model, self.epsilon, self.config, self.device, self.audio_folder)




























