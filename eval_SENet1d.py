import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

logger = logging.getLogger("add_challenge")
logger.setLevel(logging.INFO)

from torch.utils.data import DataLoader
from src.SENet.senet1d_model import se_resnet341d_custom
from src.rawnet_utils import LoadEvalData_RawNet
from src.utils import *
from tqdm import tqdm
import csv
import sys

import os



def SENet1D_eval(senet1d_model,
                save_path,
                device,
                config,
                type_of_spec,
                epsilon,
                attack_model,
                attack,
                dataset,
                feature,
                 q1, q2, q3, eps1, eps2, eps3,
                 trip):

    epsilon_dot_notation = str(epsilon).replace('.', 'dot')
    eps1_str = str(eps1).replace('.', 'dot')
    eps2_str = str(eps2).replace('.', 'dot')
    eps3_str = str(eps3).replace('.', 'dot')

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script

    if feature == 'audio':
        if attack != 'Ens1D' and attack != 'Ens2D':
            feat_directory = os.path.join(script_dir, 'attacks',
                                          f'{attack}_{attack_model}_{model_version}_{type_of_spec}',
                                          f'{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'Ens1D' and trip==0:
            feat_directory = os.path.join(script_dir, 'attacks', f'{attack}_{attack_model}_{model_version}_{type_of_spec}',
                                          f'{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{q1}_{q2}_{eps1_str}_{eps2_str}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{q1}_{q2}_{eps1_str}_{eps2_str}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'Ens1D' and trip == 1:
            feat_directory = os.path.join(script_dir, 'attacks',
                                          f'{attack}_{attack_model}_{model_version}_{type_of_spec}',
                                          f'{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{q1}_{q2}_{q3}_{eps1_str}_{eps2_str}_{eps3_str}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{q1}_{q2}_{q3}_{eps1_str}_{eps2_str}_{eps3_str}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'Ens2D':
            sys.exit('2D ensemble todo')
        else:
            sys.exit(f'Unknown type of attack {attack} on {attack_model}')



    if os.path.exists(csv_location):
        os.remove(csv_location)
        print(f"Existing file '{csv_location}' has been removed.")

        # create and write the csv file
    with open(csv_location, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # write the header
        csvwriter.writerow(['path'])
        # write the data rows
        for index, filename in enumerate(feat_files):
            csvwriter.writerow([os.path.join(feat_directory, filename)])

        # csv file done
    df_eval = pd.read_csv(csv_location)
    file_eval = list(df_eval['path'])

    if os.path.exists(save_path):
        print(f'save_path exists, removing it to create a new one')
        os.system(f'rm {save_path}')

    assert feature == 'audio', print(f'Feature can only be audio, not {feature}')

    feat_set = LoadEvalData_RawNet(list_IDs=file_eval, config=config)
    feat_loader = DataLoader(feat_set, batch_size=config['eval_batch_size'], shuffle=False, num_workers=15)

    senet1d_model.eval()

    with torch.no_grad():

        for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
            # fname_list = []
            # score_list = []
            feat_batch = feat_batch.to(torch.float32).to(device)
            score = senet1d_model(feat_batch)
            #probabilities = torch.exp(score)
            probabilities = score
            probabilities = probabilities.detach().cpu().numpy()

            with open(save_path, mode='a+', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(['Filename', 'Pred.class 0', 'Pred.class 1'])
                for i in range(len(utt_id)):
                    row = [utt_id[i], probabilities[i, 0], probabilities[i, 1]]
                    writer.writerow(row)
        print('Scores saved to {}'.format(save_path))




def init_eval(config, type_of_spec, epsilon, attack_model, model_version, attack, dataset, feature, q1, q2, q3, eps1, eps2, eps3, trip):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script

    model = se_resnet341d_custom(num_classes=2).to(device)

    if model_version != 'v0':
        sys.exit(f'Version {model_version} is not accepted')

    # load the correct model
    if type_of_spec == 'mag':
        if model_version == 'v0':
            model.load_state_dict(torch.load(os.path.join(script_dir, config['model_path_spec_mag_v0']), map_location=device), strict=False)
    elif type_of_spec == 'pow':
        if model_version == 'v0':
            model.load_state_dict(torch.load(os.path.join(script_dir, config['model_path_spec_pow_v0']), map_location=device), strict=False)
    else:
        sys.exit('Wrong type of spectrogram mode: should be pow or mag')

    if attack != 'Ens2D' and attack != 'Ens1D' and trip==0:
        epsilon_str = str(epsilon).replace('.', 'dot')
        save_path = os.path.join(script_dir,
                                 'eval',
                                 f'probs_SENet1D_{model_version}_{attack}_{attack_model}_{dataset}_{epsilon_str}_{type_of_spec}_{feature}.csv')
        SENet1D_eval(model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset,
                    feature, q1, q2, q3, eps1, eps2, eps3)

    elif attack == 'Ens1D' and trip==0:
        eps1_str = str(eps1).replace('.', 'dot')
        eps2_str = str(eps2).replace('.', 'dot')
        save_path = os.path.join(script_dir,
                         'eval',
                         f'probs_SENet1D_{model_version}_{attack}_{attack_model}_{q1}_{q2}_{eps1_str}_{eps2_str}_{type_of_spec}_{feature}.csv')
        SENet1D_eval(model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature,
                     q1, q2, q3, eps1, eps2, eps3)
    elif attack == 'Ens1D' and trip == 1:
        eps1_str = str(eps1).replace('.', 'dot')
        eps2_str = str(eps2).replace('.', 'dot')
        eps3_str = str(eps3).replace('.', 'dot')
        save_path = os.path.join(script_dir,
                                 'eval',
                                 f'probs_SENet1D_{model_version}_{attack}_{attack_model}_{q1}_{q2}_{q3}_{eps1_str}_{eps2_str}_{eps3_str}_{type_of_spec}_{feature}.csv')
        SENet1D_eval(model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature,
                      q1, q2,q3, eps1, eps2, eps3, trip=trip)

    elif attack == 'Ens2D':
        sys.exit('TODO 2D ens')
    else:
        sys.exit(f'Invalid attack combination for {attack}, {attack_model}')




if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, 'config/senet1d.yaml')
    config = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'Ens1D'  # 'FGSM' or 'Ensemble'
    attack_model = 'ResSENRaw'  #'ResNet' or 'SENet' or 'ResNet1D'
    epsilon = None
    dataset = 'whole'  # '3s' or 'whole'-
    model_version = 'v0'  # or 'old'  version of eval and attack_model
    type_of_spec = 'pow'  # 'pow' or 'mag'
    feature = 'audio'  # SeNet1d can only work with 1d inputs
    q1 = 50
    q2 = 80
    q3 = 60
    eps1 = 0.03
    eps2 = 0.008
    eps3 = 0.02
    trip = 1  # 0 if not the triplet ensemble


    init_eval(config,
              type_of_spec=type_of_spec,
              epsilon=epsilon,
              attack_model=attack_model,
              model_version=model_version,
              attack=attack,
              dataset=dataset,
              feature=feature,
              q1=q1, q2=q2, q3=q3, eps1=eps1, eps2=eps2, eps3=eps3, trip=trip)