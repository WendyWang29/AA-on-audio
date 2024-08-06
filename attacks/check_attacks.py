import csv
import logging
from src.utils import *
import matplotlib.pyplot as plt
from attacks_utils import get_mini_batch
from src.resnet_features import compute_spectrum
from src.resnet_model import SpectrogramModel
from src.rawnet2_model import RawNet
from src.rawnet_utils import get_waveform, create_mini_batch_RawNet
from torch import Tensor
import librosa
import re
import pandas as pd

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('numba').setLevel(logging.WARNING)

def load_ResNet():
    config_path = '../config/residualnet_train_config.yaml'
    config = read_yaml(config_path)
    model = SpectrogramModel().to(device)
    model.load_state_dict(torch.load(os.path.join( config["model_path_spec"]), map_location=device))
    model.eval()
    return model, config

def load_RawNet2():
    config_path = '../config/rawnet2.yaml'
    config = read_yaml(config_path)
    model = RawNet(config['model'], device)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(config['model_path_spec']), map_location=device))
    model.eval()
    return model, config

# def create_mini_batch_RawNet(audio):
#     feature_len = audio.shape[0]
#     network_input_shape = 16000 * 4
#     if feature_len < network_input_shape:
#         num_repeats = int(network_input_shape / feature_len) + 1
#         audio = np.tile(audio, num_repeats)
#     X_win = audio[: network_input_shape]
#     X_win = np.expand_dims(X_win, axis=0)
#     X_win = Tensor(X_win)
#     return X_win # the mini batch, still on CPU

def extract_id(file_path):
    match = re.search(r'LA_E_(\d+)', file_path)
    if match:
        return match.group(1)
    return None

def get_original_audio(audio_name):
    path = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac'
    og_path = os.path.join(path, 'LA_E_' + extract_id(audio_name) + '.flac')
    audio, _ = librosa.load(og_path, sr=None, duration=240, mono=True)
    return audio, og_path

def get_GT(audio_name, eval_path):
    number = extract_id(audio_name)
    data = []
    with open(eval_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append({'index': row[0], 'path': row[1], 'label': row[2]})
    # search for row containing the number
    label = None
    for row in data:
        if f'LA_E_{number}.flac' in row['path']:
            label = row['label']
            break
    if label is not None:
        return label
    else:
        pass

def plot_specs(audio, audio_name):
    spec = compute_spectrum(audio) #perturbed spec
    og_audio, og_path = get_original_audio(audio_name)
    og_spec = compute_spectrum(og_audio)
    sr = 16000

    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    librosa.display.specshow(spec, x_axis='time', sr=sr, y_axis='linear')
    plt.title(f'Power spectrogram for\n {audio_name}', fontsize=8)
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(og_spec, x_axis='time', sr=sr, y_axis='linear')
    plt.title(f'Power spectrogram for\n {og_path}', fontsize=8)
    plt.colorbar(format='%+2.0f dB')

    plt.subplots_adjust(hspace=0.7)
    plt.tight_layout()
    plt.show()

def get_predicted_label(out):
    if out[0,0] > out[0,1]:
        return 0
    else:
        return 1

def check_audio_given_the_name(audio_name, model_to_use, epsilon, config):
    # name is like /FGSM_RawNet_LA_E_2834763_0dot005.flac
    epsilon_str = str(epsilon).replace('.', 'dot')
    eval_path = os.path.join('..', config['df_eval_path'])

    if model_to_use == 'ResNet':

        path = os.path.join('FGSMc_data', f'FGSMc_ResNet_dataset_{epsilon_str}', audio_name)
        #path = os.path.join('BIMCut_data', f'BIMCut_ResNet_dataset_{epsilon_str}', audio_name)
        #path = os.path.join('FGSM_data', f'FGSM_dataset_{epsilon_str}', audio_name)
        #path = os.path.join('BIM_data', f'BIM_RawNet_dataset_{epsilon_str}', audio_name)
        #path = os.path.join('SSA_data', f'SSA_ResNet_dataset_{epsilon_str}', audio_name)
        #path = os.path.join('FGSMS_data', f'FGSMS_RawNet_dataset_{epsilon_str}', audio_name)

        audio = get_waveform(path, config)
        # temp = int(len(audio)/2)
        # audio = audio[:temp]
        spec = compute_spectrum(audio)

        plot_specs(audio, audio_name)

        spec_length = spec.shape[1]
        net_input_shape = 28 * 3
        if spec_length < net_input_shape:
            num_repeats = int(net_input_shape / spec_length) + 1
            spec = np.tile(spec, (1, num_repeats))
        spec = spec[:, :net_input_shape]
        spec_batch = get_mini_batch(spec, device).to(device)

        out = model(spec_batch)
        probabilities = torch.exp(out)
        gt_label = get_GT(audio_name, eval_path)
        print(f'File: {audio_name}\n'
              f'The GT label is {gt_label}\n'
              f'The predicted label is {get_predicted_label(probabilities)}')

    elif model_to_use == 'RawNet2':

        #path = os.path.join('DeepFool_RawNet', f'DeepFool_RawNet_dataset', audio_name)
        #path = os.path.join('FGSM_RawNet_data', f'FGSM_RawNet_dataset_{epsilon_str}', audio_name)
        path = os.path.join('BIM_data', f'BIM_RawNet_dataset_{epsilon_str}', audio_name)
        #path = os.path.join('PGD_data', f'PGD_RawNet_dataset_{epsilon_str}', audio_name)
        #path = os.path.join('SSA_data', f'SSA_ResNet_dataset_{epsilon_str}', audio_name)
        #path = os.path.join('FGSMS_data', f'FGSMS_RawNet_dataset_{epsilon_str}', audio_name)

        audio = get_waveform(path, config)
        # temp = int(len(audio)/2)
        # audio = audio[:temp]
        plot_specs(audio, audio_name)
        audio_batch = create_mini_batch_RawNet(audio).to(device)
        out = model(audio_batch)
        probabilities = torch.exp(out)
        gt_label = get_GT(audio_name, eval_path)
        print(f'File: {audio_name}\n'
              f'The GT label is {gt_label}\n'
              f'The predicted label is {get_predicted_label(probabilities)}')






if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #______ THINGS TO SET ______#
    model_to_use = 'RawNet2'
    #model_to_use = 'ResNet'
    epsilon = 0.005
    audio_name = 'BIM_RawNet_LA_E_5849185_0dot005.flac'
    #__________________________#

    if model_to_use == 'ResNet':
        model, config = load_ResNet()
    elif model_to_use == 'RawNet2':
        model, config = load_RawNet2()

    print(f'Model being evaluated: {model_to_use}\n'
          f'File name: {audio_name}\n'
          f'Epsilon: {epsilon}')

    check_audio_given_the_name(audio_name=audio_name,
                               model_to_use=model_to_use,
                               epsilon=epsilon,
                               config=config)




