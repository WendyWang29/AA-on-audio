import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

logger = logging.getLogger("add_challenge")
logger.setLevel(logging.INFO)
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.SENet.senet1d_model import se_resnet341d_custom
from src.rawnet_utils import LoadEvalData_RawNet
from src.utils import *
from tqdm import tqdm
import csv
import sys
import re
from sklearn import model_selection
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
                q_res,
                q_sen):

    epsilon_dot_notation = str(epsilon).replace('.', 'dot')

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script

    if feature == 'audio':
        if attack != 'Ensemble' and attack != 'BIM' and attack != 'Ensemble1D' and attack != 'Ensemble1D_RaS' and attack != 'Ensemble1D_RS':
            feat_directory = os.path.join(script_dir, 'attacks', f'{attack}_{attack_model}_{model_version}_{type_of_spec}',
                                          f'{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'BIM':
            feat_directory = os.path.join(script_dir, 'attacks',
                                          f'{attack}_{attack_model}_{model_version}_{type_of_spec}',
                                          f'{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'Ensemble':
            feat_directory = os.path.join(script_dir, 'attacks', 'Ensemble',
                                          f'QUANT_ENS_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_Ensemble_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'Ensemble1D':
            feat_directory = os.path.join(script_dir, 'attacks', 'Ensemble1D',
                                          f'QUANT_ENS1D_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_Ensemble1D_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'Ensemble1D_RS':
            feat_directory = os.path.join(script_dir, 'attacks', 'Ensemble1D_RS',
                                          f'QUANT_ENS1D_RS_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_Ensemble1D_RS_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'Ensemble1D_RaS':
            feat_directory = os.path.join(script_dir, 'attacks', 'Ensemble1D_RaS',
                                          f'QUANT_ENS1D_RaS_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_Ensemble1D_RaS_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
    else:
        sys.exit('This model can only handle as input audio files...')



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
            probabilities = torch.exp(score)
            probabilities = probabilities.detach().cpu().numpy()

            with open(save_path, mode='a+', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(['Filename', 'Pred.class 0', 'Pred.class 1'])
                for i in range(len(utt_id)):
                    row = [utt_id[i], probabilities[i, 0], probabilities[i, 1]]
                    writer.writerow(row)
        print('Scores saved to {}'.format(save_path))




def init_eval(config, type_of_spec, epsilon, attack_model, model_version, attack, dataset, feature, q_res, q_sen):

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


    if attack != 'Ensemble' and attack != 'Ensemble1D' and attack != 'Ensemble1D_RS' and attack != 'Ensemble1D_RaS':
        epsilon_str = str(epsilon).replace('.', 'dot')
        save_path = os.path.join(script_dir,
                                 'eval',
                                 f'probs_SENet1D_{model_version}_{attack}_{attack_model}_{dataset}_{epsilon_str}_{type_of_spec}_{feature}.csv')
        SENet1D_eval(model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset,
                    feature, q_res, q_sen)


    elif attack == 'Ensemble' or attack == 'Ensemble1D' or attack == 'Ensemble1D_RS' or attack == 'Ensemble1D_RaS':
        epsilon_str = str(epsilon).replace('.', 'dot')
        save_path = os.path.join(script_dir,
                                 'eval',
                                 f'probs_SENet1D_{model_version}_{attack}_{dataset}_{q_res}_{q_sen}_{epsilon_str}_{type_of_spec}_{feature}.csv')
        SENet1D_eval(model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen)


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
    attack = 'BIM'  # 'FGSM' or 'Ensemble'
    attack_model = 'SENet1D'  #'ResNet' or 'SENet' or 'ResNet1D'
    epsilon = None
    dataset = 'whole'  # '3s' or 'whole'
    model_version = 'v0'  # or 'old'  version of eval and attack_model
    type_of_spec = 'pow'  # 'pow' or 'mag'
    feature = 'audio'  # SeNet1d can only work with 1d inputs
    q_res = 10  # first model
    q_sen = 10  # second model



    init_eval(config,
              type_of_spec=type_of_spec,
              epsilon=epsilon,
              attack_model=attack_model,
              model_version=model_version,
              attack=attack,
              dataset=dataset,
              feature=feature,
              q_res=q_res,
              q_sen=q_sen)