import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

logger = logging.getLogger("add_challenge")
logger.setLevel(logging.INFO)
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.resnet_model import SpectrogramModel
from src.resnet_utils import LoadEvalData_ResNet, LoadEvalData_ResNet_SPEC
from src.utils import *
from tqdm import tqdm
import csv
import sys
import re
from sklearn import model_selection
import os



def ResNet_eval(resnet_model,
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
    model_version = 'v0'
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script

    if feature == 'audio':
        if attack != 'Ensemble' and attack != 'Ensemble1D' and attack != 'Ensemble1D_RS' and attack != 'Ensemble1D_RaS':
            feat_directory = os.path.join(script_dir, 'attacks', f'{attack}_{attack_model}_{model_version}_{type_of_spec}',
                                          f'{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_flac_{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        elif attack == 'Ensemble':
            assert attack == 'Ensemble', print('Wrong attack')
            feat_directory = os.path.join(script_dir, 'attacks', 'Ensemble', f'QUANT_ENS_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
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


    elif feature == 'spec':
        if attack != 'Ensemble' and attack != None:
            feat_directory = os.path.join(script_dir, 'attacks', f'{attack}_{attack_model}_{model_version}_{type_of_spec}',
                                          f'{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}', 'spec')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_spec_{attack}_{attack_model}_{model_version}_{dataset}_{type_of_spec}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.npy')]
        else:
            assert attack == 'Ensemble', print('Wrong attack')
            feat_directory = os.path.join(script_dir, 'attacks', 'Ensemble',
                                          f'QUANT_ENS_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}', 'spec')
            csv_location = os.path.join(script_dir, 'eval',
                                        f'list_spec_Ensemble_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_dot_notation}')
            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.npy')]


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



    if feature == 'spec':
        feat_set = LoadEvalData_ResNet_SPEC(list_IDs=file_eval, win_len=config_res['win_len'], config=config,
                                            type_of_spec=type_of_spec)
    elif feature == 'audio':
        feat_set = LoadEvalData_ResNet(list_IDs=file_eval, win_len=config_res['win_len'], config=config,
                                            type_of_spec=type_of_spec)
    else:
        sys.exit('Wrong type of feature, should be spec or audio')

    feat_loader = DataLoader(feat_set, batch_size=config_res['eval_batch_size'], shuffle=False, num_workers=10)

    resnet_model.eval()

    with torch.no_grad():

        for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
            # fname_list = []
            # score_list = []
            feat_batch = feat_batch.to(torch.float32).to(device)
            score = resnet_model(feat_batch)
            probabilities = score
            #probabilities = torch.exp(score)
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

    resnet_model = SpectrogramModel().to(device)

    if model_version != 'v0':
        sys.exit(f'Version {model_version} is not accepted')

    # load the correct model
    if type_of_spec == 'mag':
        if model_version == 'v0':
            resnet_model.load_state_dict(torch.load(os.path.join(script_dir, config['model_path_spec_mag']), map_location=device), strict=False)
    elif type_of_spec == 'pow':
        if model_version == 'v0':
            resnet_model.load_state_dict(torch.load(os.path.join(script_dir, config['model_path_spec_pow_v0']), map_location=device), strict=False)
    else:
        sys.exit('Wrong type of spectrogram mode: should be pow or mag')


    if attack != 'Ensemble' and attack != 'Ensemble1D' and attack != 'Ensemble1D_RS' and attack != 'Ensemble1D_RaS':
        epsilon_str = str(epsilon).replace('.', 'dot')
        save_path = os.path.join(script_dir,
                                 'eval',
                                 f'probs_{eval_model}_{model_version}_{attack}_{attack_model}_{dataset}_{epsilon_str}_{type_of_spec}_{feature}.csv')
        ResNet_eval(resnet_model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset,
                    feature, q_res, q_sen)


    elif attack == 'Ensemble' or attack == 'Ensemble1D' or attack == 'Ensemble1D_RS' or attack == 'Ensemble1D_RaS':
        epsilon_str = str(epsilon).replace('.', 'dot')
        save_path = os.path.join(script_dir,
                                 'eval',
                                 f'probs_{eval_model}_{model_version}_{attack}_{dataset}_{q_res}_{q_sen}_{epsilon_str}_{type_of_spec}_{feature}.csv')
        ResNet_eval(resnet_model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen)

    else:
        sys.exit(f'Invalid attack combination for {attack}, {attack_model}')




if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-2)

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, 'config/residualnet_train_config.yaml')
    config_res = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'BIM'  # 'FGSM' or 'Ensemble'
    attack_model = 'ResNet2D'  #'ResNet' or 'SENet'
    eval_model = 'ResNet2D'
    epsilon = 3.0
    dataset = 'whole'  # '3s' or 'whole'
    model_version = 'v0'  # or 'old'  version of eval and attack_model
    type_of_spec = 'pow'  # 'pow' or 'mag'
    feature = 'audio'  #'spec' or 'audio'
    q_res = 10   # model1
    q_sen = 10   # model2

    '''
    #######################################
    '''

    init_eval(config_res,
              type_of_spec=type_of_spec,
              epsilon=epsilon,
              attack_model=attack_model,
              model_version=model_version,
              attack=attack,
              dataset=dataset,
              feature=feature,
              q_res=q_res,
              q_sen=q_sen)

