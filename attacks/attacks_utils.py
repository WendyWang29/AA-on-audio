"""
ATTACK SCRIPTS
"""
import torch
import os
import pandas as pd
import numpy as np
import torch
import soundfile as sf
import torch.nn as nn
from src.resnet_model import SpectrogramModel
from src.resnet_utils import get_features
from sp_utils import spectrogram_inversion, get_spectrogram_from_audio



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
    p_spec = p_spec.squeeze(0).detach()
    p_spec = p_spec.cpu()
    p_spec = p_spec.numpy()

    return p_spec

def spec_to_tensor(spec, device):
    X_p_batch = np.expand_dims(spec, axis=0)  # ndarray
    X_p_batch_tensor = torch.from_numpy(X_p_batch).to(device)  # tensor
    X_p = X_p_batch_tensor
    X_p.requires_grad = True
    return X_p

def get_class(out):
    score_p = out[0, 0] - out[0, 1]
    if score_p > 0:
        pred_p = 0
    else:
        pred_p = 1
    return pred_p

def save_perturbed_audio(file, folder, audio, sr, epsilon, attack):
    # ensure folder path exists
    os.makedirs(folder, exist_ok=True)

    # create the file path
    file_name = os.path.splitext(os.path.basename(file))[0]
    file_path = os.path.join(folder, f'{attack}_{file_name}_{epsilon}.flac')

    # check if the same file already exists. If yes remove the old one
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Removed existing file: {file_path}')

    sf.write(file_path, audio, sr, format='FLAC')
    print(f'Saved the perturbed audio as: {file_path}')

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
    # ensure folder path exists
    os.makedirs(folder, exist_ok=True)

    # create the file path
    file_name = os.path.splitext(os.path.basename(file))[0]
    file_path = os.path.join(folder, f'{attack}_{file_name}_{epsilon}.npy')

    # check if the same file already exists. If yes remove the old one
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Removed existing file: {file_path}')

    np.save(file_path, spec)
    print(f'Saved the perturbed spec as: {file_path}')

def evaluate_spec(spec, model, device):
    spec = spec_to_tensor(spec,device)
    out = model(spec)
    return out


class Attack:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device

    def attack_single(self, index):
        # given the index retrieve the file path, the GT label and the spectrogram
        file, label, spec = retrieve_single_cached_spec(index=index,
                                                        config=self.config)
        # turn the single spec into a mini-batch to be fed to the model
        spec = get_mini_batch(spec, self.device)
        return spec, label, file

    def evaluate_single(self, label, perturbed_spec, audio_path, perturbed_audio):
        out_spec = evaluate_spec(perturbed_spec, self.model, self.device)
        print(f'Model output for the perturbed spectrogram is: {out_spec}')
        print(f'The predicted class is {get_class(out_spec)} and the GT label is {label}')

        perturbed_audio_spec = get_spectrogram_from_audio(audio_path)
        out_audio = evaluate_spec(perturbed_audio_spec, self.model, self.device)
        print(f'Model output for the perturbed audio is: {out_audio}')
        print(f'The predicted class is {get_class(out_audio)} and the GT label is {label}')



    def attack_batch(self):
        pass


class FGSMAttack(Attack):
    def __init__(self, epsilon, config, model, device):
        super().__init__(config, model, device)
        self.epsilon = epsilon

    def attack_single(self, index):
        spec, label, file = super().attack_single(index)

        attack = 'FGSM'
        # define the paths for saving the perturbed data
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.folder_specs_FGSM = os.path.join(self.current_dir, 'FGSM_data', 'p_specs')
        self.folder_audio_FGSM = os.path.join(self.current_dir, 'FGSM_data', 'p_audio')

        # run the FGSM attack on the single spectrogram
        perturbed_spec = FGSM_perturb(spec, self.model, self.epsilon, label, self.device)

        # retrieve the perturbed audio
        perturbed_audio = spectrogram_inversion(perturbed_spec)

        # save the files
        save_perturbed_spec(file=file,
                            folder=self.folder_specs_FGSM,
                            spec=perturbed_spec,
                            epsilon=self.epsilon,
                            attack=attack)

        save_perturbed_audio(file=file,
                             folder=self.folder_audio_FGSM,
                             audio=perturbed_audio,
                             sr=16000,
                             epsilon=self.epsilon,
                             attack=attack)

        self.evaluate_single(label, perturbed_spec, file, perturbed_audio)
























