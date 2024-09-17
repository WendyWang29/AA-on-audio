import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

logger = logging.getLogger("add_challenge")
logger.setLevel(logging.INFO)
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.LCNN_model.LCNN_model import LCNN
from src.resnet_utils import LoadEvalData_ResNet, LoadEvalData_ResNet_SPEC
from src.utils import *
from tqdm import tqdm
import csv
import sys
import re
from sklearn import model_selection
import os

def LCNN_eval(lcnn_model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen):
    epsilon_dot_notation = str(epsilon).replace('.', 'dot')

    if dataset != '3s':
        if feature == 'audio' and attack != 'Ensemble':
            feat_directory = os.path.join('attacks', f'{attack}_{attack_model}',
                                          f'{attack}_{attack_model}_dataset_{epsilon_dot_notation}')
            csv_location = os.path.join('eval', f'flac_{attack}_{attack_model}_{epsilon_dot_notation}.csv')

            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]

        elif feature == 'audio' and attack == 'Ensemble':
            feat_directory = os.path.join('attacks', 'Ensemble',
                                          f'QUANT_ENS_{q_res}_{q_sen}_{epsilon_dot_notation}_dataset')
            csv_location = os.path.join('eval', f'flac_Ensemble_{q_res}_{q_sen}_{epsilon_dot_notation}.csv')

            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]

        elif feature == 'spec' and attack == 'Ensemble':
            pass
            #TODO

        elif attack == None:
            pass
            #TODO

    elif dataset == '3s':
        if feature == 'spec' and attack != 'Ensemble':
            feat_directory = os.path.join('attacks', f'{attack}_3s_{attack_model}',
                                          f'{attack}_{attack_model}_3s_dataset_{epsilon_dot_notation}',
                                          'spec')
            csv_location = os.path.join('eval', f'spec_{attack}_3s_{attack_model}_{epsilon_dot_notation}.csv')

            # create list of npy files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.npy')]

        elif feature == 'audio' and attack != 'Ensemble':
            feat_directory = os.path.join('attacks', f'{attack}_3s_{attack_model}',
                                          f'{attack}_{attack_model}_3s_dataset_{epsilon_dot_notation}')
            csv_location = os.path.join('eval', f'flac_{attack}_3s_{attack_model}_{epsilon_dot_notation}.csv')

            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]

        elif feature == 'audio' and attack == 'Ensemble':
            feat_directory = os.path.join('attacks', 'Ensemble', f'QUANT_ENS_{q_res}_{q_sen}_{epsilon_dot_notation}_dataset')
            csv_location = os.path.join('eval', f'flac_{attack}_{q_res}_{q_sen}_{epsilon_dot_notation}.csv')

            # create list of flac files
            feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]


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

    feat_loader = DataLoader(feat_set, batch_size=config_res['eval_batch_size'], shuffle=False, num_workers=15)

    lcnn_model.eval()

    with torch.no_grad():

        for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
            # fname_list = []
            # score_list = []
            feat_batch = feat_batch.to(torch.float32).to(device)
            score = lcnn_model(feat_batch)
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




def init_eval(config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lcnn_model = LCNN().to(device)
    if type_of_spec == 'mag':
        lcnn_model.load_state_dict(torch.load(config['model_path_spec_mag'], map_location=device), strict=False)
    elif type_of_spec == 'pow':
        lcnn_model.load_state_dict(torch.load(config['model_path_spec_pow'], map_location=device), strict=False)
    else:
        sys.exit('Wrong type of spectrogram mode: should be pow or mag')



    if attack == None:
        # perform the evaluation on the clean dataset ASVSpoof2019
        df_eval = pd.read_csv(config['df_eval_path'])
        if type_of_spec == 'mag':
            save_path = 'eval/prob_LCNN_spec_eval_mag.csv'
        elif type_of_spec == 'pow':
            save_path = 'eval/prob_LCNN_spec_eval.csv'
        LCNN_eval(lcnn_model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen)

    elif attack != 'Ensemble':
        if feature == 'audio' and dataset != '3s':
            epsilon_str = str(epsilon).replace('.', 'dot')
            save_path = f'eval/prob_LCNN_{attack}_{attack_model}_{epsilon_str}.csv'
            LCNN_eval(lcnn_model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen )
        elif feature == 'audio' and dataset == '3s':
            epsilon_str = str(epsilon).replace('.', 'dot')
            save_path = f'eval/prob_LCNN_{attack}_{attack_model}_3s_{epsilon_str}_AUDIO.csv'
            LCNN_eval(lcnn_model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen)
        elif feature == 'spec' and dataset == '3s':
            epsilon_str = str(epsilon).replace('.', 'dot')
            save_path = f'eval/prob_LCNN_{attack}_{attack_model}_3s_{epsilon_str}_SPEC.csv'
            LCNN_eval(lcnn_model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen)

    elif attack == 'Ensemble':
        epsilon_str = str(epsilon).replace('.', 'dot')
        save_path = f'eval/prob_LCNN_Ensemble_{q_res}_{q_sen}_{epsilon_str}.csv'
        LCNN_eval(lcnn_model, save_path, device, config, type_of_spec, epsilon, attack_model, attack, dataset, feature, q_res, q_sen)

    else:
        sys.exit(f'Invalid attack combination for {attack}, {attack_model}')





if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)

    config_path = 'config/LCNN.yaml'
    config_res = read_yaml(config_path)


    type_of_spec = 'pow'
    epsilon = 3.0
    attack_model = None
    attack = 'Ensemble'
    dataset = '3s'
    feature = 'audio' #'audio' or 'spec'
    q_res = 10
    q_sen = 10

    init_eval(config_res,
              type_of_spec=type_of_spec,
              epsilon=epsilon,
              attack_model=attack_model,
              attack=attack,
              dataset=dataset,
              feature=feature,
              q_res=q_res,
              q_sen=q_sen)
