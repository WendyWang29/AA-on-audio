import csv

from src.utils import *
from src.resnet_model import SpectrogramModel
from src.rawnet2_model import RawNet
from src.rawnet_utils import get_waveform
from torch import Tensor
import re
import pandas as pd



def load_ResNet():
    config_path = '../config/residualnet_train_config.yaml'
    config = read_yaml(config_path)
    model = SpectrogramModel().to(device)
    model.load_state_dict(torch.load(os.path.join('..', config["model_path_spec"]), map_location=device))
    model.eval()
    return model, config

def load_RawNet2():
    config_path = '../config/rawnet2.yaml'
    config = read_yaml(config_path)
    model = RawNet(config['model'], device)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec']), map_location=device))
    model.eval()
    return model, config

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

def extract_id(file_path):
    match = re.search(r'LA_E_(\d+)', file_path)
    if match:
        return match.group(1)
    return None


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

def get_predicted_label(out):
    if out[0,0] > out[0,1]:
        return 0
    else:
        return 1

def check_audio_given_the_name(audio_name, model_to_use, epsilon, config):
    # name is like /FGSM_RawNet_LA_E_2834763_0dot005.flac
    epsilon_str = str(epsilon).replace('.', 'dot')
    #df_eval = pd.read_csv(os.path.join('..', config['df_eval_path']), header=0)
    eval_path = os.path.join('..', config['df_eval_path'])

    if model_to_use == 'ResNet':
        pass
    elif model_to_use == 'RawNet2':
        path = os.path.join('FGSM_data', f'FGSM_RawNet_dataset_{epsilon_str}', audio_name)
        audio = get_waveform(path, config)
        audio_batch = create_mini_batch_RawNet(audio).to(device)
        out = model(audio_batch)
        probabilities = torch.exp(out)
        gt_label = get_GT(audio_name, eval_path)
        print(f'File: {audio_name}\n'
              f'The GT label is {gt_label}\n'
              f'The predicted label is {get_predicted_label(out)}')




if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model_to_use = 'ResNet'
    model_to_use = 'RawNet2'
    epsilon = 0.005

    if model_to_use == 'ResNet':
        model, config = load_ResNet()
    elif model_to_use == 'RawNet2':
        model, config = load_RawNet2()

    check_audio_given_the_name(audio_name='FGSM_RawNet_LA_E_2834763_0dot005.flac',
                               model_to_use=model_to_use,
                               epsilon=epsilon,
                               config=config)



