import sys
from src.utils import *
import csv
import re
import librosa
from src.resnet_features import compute_spectrum
from src.resnet_model import SpectrogramModel
from src.LCNN_model.LCNN_model import LCNN
from src.SENet.SENet_model import se_resnet34_custom

def load_resnet(device):
    config_path = '../config/residualnet_train_config.yaml'
    config = read_yaml(config_path)
    model = SpectrogramModel().to(device)
    model.load_state_dict(torch.load(os.path.join('..', config["model_path_spec"]), map_location=device))
    model.eval()
    return model, config

def load_lcnn(device):
    config_path = '../config/LCNN.yaml'
    config = read_yaml(config_path)
    model = LCNN().to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec']), map_location=device))
    model.eval()
    return model, config

def load_senet(device):
    config_path = '../config/SENet.yaml'
    config = read_yaml(config_path)
    model = se_resnet34_custom(num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec']), map_location=device))
    model.eval()
    return model, config

def get_original_audio(file_number):
    path = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac'
    file_number = str(file_number)
    original_path = os.path.join(path, f'LA_E_{file_number}' + '.flac')
    audio, _ = librosa.load(original_path, sr=None, duration=240, mono=True)
    return audio, original_path

def get_GT_label(file_number, eval_path):
    data = []
    with open(eval_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append({'index': row[0], 'path': row[1], 'label': row[2]})
    # search for row containing the number
    label = None
    for row in data:
        if f'LA_E_{file_number}.flac' in row['path']:
            label = row['label']
            break
    if label is not None:
        return label
    else:
        pass

def get_model_prediction(eval_model, pert_audio, device, flag=None):
    spec = compute_spectrum(pert_audio)
    spec_length = spec.shape[1]
    net_input_shape = 28 * 3
    if spec_length < net_input_shape:
        num_repeats = int(net_input_shape / spec_length) + 1
        spec = np.tile(spec, (1, num_repeats))
    spec = spec[:, :net_input_shape]
    spec_batch = np.expand_dims(spec, axis=0)
    spec_batch = torch.from_numpy(spec_batch).to(device)

    if flag == 0:
        out = eval_model(spec_batch)
    elif flag == 1:
        # model is SENet
        out = eval_model(spec_batch.unsqueeze(dim=1))

    probabilities = torch.exp(out)
    pred_label = torch.argmax(probabilities)
    return out, pred_label, spec

def get_og_spec(original_audio):
    spec = compute_spectrum(original_audio)
    spec_length = spec.shape[1]
    net_input_shape = 28 * 3
    if spec_length < net_input_shape:
        num_repeats = int(net_input_shape / spec_length) + 1
        spec = np.tile(spec, (1, num_repeats))
    spec = spec[:, :net_input_shape]
    return spec

def compute_confidence(out):
    probs = torch.exp(out)
    probs = probs / probs.sum()
    confidence = probs.max().item()
    confidence_perc = confidence * 100
    return confidence_perc



def check_attack(eval_model, attack_model, attack, file_number, epsilon, device):
    '''
    Check one perturbed (attacked) audio
    :param eval_model: model on which we want to test the perturbed audio
    :param attack_model: model which had been used to craft the attack
    :param attack: attack used (FGSM or BIM)
    :param file_number: file identifying number
    :param epsilon: epsilon used
    :param device: device
    :return: perturbed file name, perturbed audio, original audio
    '''
    flag = 0
    string = eval_model

    # load the evaluation model
    if eval_model == 'ResNet' and attack == 'FGSM':
        eval_model, config = load_resnet(device)
    elif eval_model == 'RawNet':
        pass
    elif eval_model == 'LCNN':
        eval_model, config = load_lcnn(device)
    elif eval_model == 'SENet':
        eval_model, config = load_senet(device)
        flag = 1  # to identify SENet model
    else:
        print('Invalid evaluation model')
        sys.exit(1)

    epsilon_str = str(epsilon).replace('.', 'dot')
    eval_path = os.path.join('..', config['df_eval_path'])  #eval dataset of ASVSpoof2019

    # set the path to the perturbed dataset
    if attack == 'FGSM' and attack_model == 'ResNet':
        folder = os.path.join('FGSM_data', f'FGSM_dataset_{epsilon_str}')
        pert_file = f'FGSM_LA_E_{file_number}_{epsilon_str}.flac'
        file_path = os.path.join(folder, pert_file)
    else:
        folder = os.path.join(f'{attack}_{attack_model}', f'{attack}_{attack_model}_dataset_{epsilon_str}')
        pert_file = f'{attack}_{attack_model}_LA_E_{file_number}_{epsilon_str}.flac'
        file_path = os.path.join(folder, pert_file)


    original_audio, _ = get_original_audio(file_number)
    original_spec = get_og_spec(original_audio)
    perturbed_audio, _ = librosa.load(file_path, sr=None, duration=240, mono=True)

    # evaluate the file
    GT_label = get_GT_label(file_number, eval_path)
    out, predicted_label, perturbed_spec = get_model_prediction(eval_model, perturbed_audio, device, flag)
    predicted_label = str(predicted_label.item())

    print(f'--> File name: {pert_file}\n'
          f'--> Model evaluated: {string}\n'
          f'--> Attack: {attack} on {attack_model}\n'
          f'--> Epsilon: {epsilon}\n'
          f'--> GT label: {GT_label}\n',
          f'--> Predicted label: {predicted_label}, \n{out}\n'
          f'--> Confidence: {compute_confidence(out):.2f} %')



    return perturbed_audio, original_audio, perturbed_spec, original_spec


def extract_id(file_path):
    match = re.search(r'LA_E_(\d+)', file_path)
    if match:
        return match.group(1)
    return None


def pred_probabilities(file2_path):
    # read df_eval_19
    file1_path = '../data/df_eval_19.csv'

    file1_ids = []
    with open(file1_path, 'r') as file1:
        csv_reader = csv.reader(file1)
        for row in csv_reader:
            file_id = extract_id(row[1])
            if file_id:
                file1_ids.append(file_id)

    # read second file and store data in a dictionary
    file2_data = {}
    with open(file2_path, 'r') as file2:
        csv_reader = csv.reader(file2)
        for row in csv_reader:
            file_id = extract_id(row[0])
            if file_id:
                file2_data[file_id] = (float(row[1]), float(row[2]))

    output_array = []
    for file_id in file1_ids:
        if file_id in file2_data:
            col2, col3 = file2_data[file_id]
            output_array.append(0 if col2 > col3 else 1)

    return output_array

# if __name__ == '__main__':
#     file = '../eval/prob_SENet_FGSM_SENet_1dot0.csv'
#     pred = pred_probabilities(file)