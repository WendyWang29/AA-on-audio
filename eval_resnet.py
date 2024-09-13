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
from src.resnet_utils import LoadEvalData_ResNet
from src.utils import *
from tqdm import tqdm
import csv
import re
from sklearn import model_selection
import os

def create_csv(attack, at_model, epsilon):
    '''
    Create a csv file with path to each flac file in attacked dataset
    :param attack: attack used
    :param at_model: model used to perform the attack
    :param epsilon: epsilon value of the attack
    :return: path to csv file
    '''

    epsilon_dot_notation = str(epsilon).replace('.', 'dot')

    # if attack == 'FGSM' and at_model == 'ResNet':
    #     # path to the directory containing the perturbed flac files
    #     flac_directory = os.path.join('attacks', f'{attack}_data', f'{attack}_dataset_{epsilon_dot_notation}')
    #     # specify full path in which csv file has to be saved
    #     csv_location = os.path.join('eval', f'flac_{attack}_{at_model}_{epsilon_dot_notation}.csv')
    # else:
    #     flac_directory = os.path.join('attacks', f'{attack}_{at_model}', f'{attack}_{at_model}_dataset_{epsilon_dot_notation}')
    #     csv_location = os.path.join('eval', f'flac_{attack}_{at_model}_{epsilon_dot_notation}.csv')

    flac_directory = os.path.join('attacks', f'{attack}_{at_model}', f'{attack}_{at_model}_dataset_{epsilon_dot_notation}')
    # specify full path in which csv file has to be saved
    csv_location = os.path.join('eval', f'flac_{attack}_{at_model}_{epsilon_dot_notation}.csv')

    if os.path.exists(csv_location):
        os.remove(csv_location)
        print(f"Existing file '{csv_location}' has been removed.")

        # create list of flac files
    flac_files = [f for f in os.listdir(flac_directory) if f.endswith('.flac')]

    # create and write the csv file
    with open(csv_location, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # write the header
        csvwriter.writerow(['path'])
        # write the data rows
        for index, filename in enumerate(flac_files):
            csvwriter.writerow([os.path.join(flac_directory, filename)])

    return csv_location

def ResNet_eval(model, save_path, config, device, attack, at_model, type_of_spec, epsilon=None, df_eval=None):

    if epsilon != None:
        # attack case --> create list of evaluation files from the attacked dataset
        path_to_csv = create_csv(attack, at_model, epsilon)
        df_eval = pd.read_csv(path_to_csv)
        file_eval = list(df_eval['path'])
    elif epsilon == None:
        # clean case --> use the clean eval dataset
        file_eval = list(df_eval['path'])
    else:
        print('There is something wrong')

    if os.path.exists(save_path):
        print(f'save_path exists, removing it to create a new one')
        os.system(f'rm {save_path}')

    feat_set = LoadEvalData_ResNet(list_IDs=file_eval, win_len=config_res['win_len'], config=config, type_of_spec=type_of_spec)
    feat_loader = DataLoader(feat_set, batch_size=config_res['eval_batch_size'], shuffle=False, num_workers=15)

    model.eval()

    with torch.no_grad():

        for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
            # fname_list = []
            # score_list = []
            feat_batch = feat_batch.to(torch.float32).to(device)
            score = model(feat_batch)
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


def ResNet_eval_ensemble(model, save_path, config, device, type_of_spec, epsilon=None, df_eval=None):
    epsilon_dot_notation = str(epsilon).replace('.', 'dot')

    # folder in which the perturbed audio files are located + create a list of the files
    flac_directory = os.path.join('attacks', 'Ensemble', f'QUANT_ENS_10_10_{epsilon_dot_notation}_dataset')
    csv_location = os.path.join('eval', f'flac_Ensemble_10_10_{epsilon_dot_notation}.csv')

    if os.path.exists(csv_location):
        os.remove(csv_location)
        print(f"Existing file '{csv_location}' has been removed.")

        # create list of flac files
    flac_files = [f for f in os.listdir(flac_directory) if f.endswith('.flac')]

    # create and write the csv file
    with open(csv_location, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # write the header
        csvwriter.writerow(['path'])
        # write the data rows
        for index, filename in enumerate(flac_files):
            csvwriter.writerow([os.path.join(flac_directory, filename)])

    # csv file done
    df_eval = pd.read_csv(csv_location)
    file_eval = list(df_eval['path'])

    if os.path.exists(save_path):
        print(f'save_path exists, removing it to create a new one')
        os.system(f'rm {save_path}')

    feat_set = LoadEvalData_ResNet(list_IDs=file_eval, win_len=config_res['win_len'], config=config,
                                   type_of_spec=type_of_spec)
    feat_loader = DataLoader(feat_set, batch_size=config_res['eval_batch_size'], shuffle=False, num_workers=15)

    model.eval()

    with torch.no_grad():

        for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
            # fname_list = []
            # score_list = []
            feat_batch = feat_batch.to(torch.float32).to(device)
            score = model(feat_batch)
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

def init_eval(config, type_of_spec, attack=None, at_model=None, epsilon=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resnet_model = SpectrogramModel().to(device)
    if type_of_spec == 'mag':
        resnet_model.load_state_dict(torch.load(config['model_path_spec_mag'], map_location=device), strict=False)
    elif type_of_spec == 'pow':
        resnet_model.load_state_dict(torch.load(config['model_path_spec_pow'], map_location=device), strict=False)

    print("Number of layers:", len(list(resnet_model.modules())))
    total_params = sum(p.numel() for p in resnet_model.parameters())
    print("Total number of parameters:", total_params)

    if attack == 'FGSM':
        # perform the evaluation on the dataset attacked with FGSM on ResNet and a certain epsilon value
        epsilon_str = str(epsilon).replace('.', 'dot')
        save_path = f'./eval/prob_ResNet_{attack}_{at_model}_{epsilon_str}.csv'
        ResNet_eval(resnet_model, save_path, config, device, attack, at_model, type_of_spec, epsilon, df_eval=None)

    elif attack == None:
        # perform the evaluation on the clean dataset ASVSpoof2019
        df_eval = pd.read_csv(config['df_eval_path'])
        if type_of_spec == 'mag':
            save_path = './eval/prob_ResNet_spec_eval_mag.csv'
        elif type_of_spec == 'pow':
            save_path = './eval/prob_ResNet_spec_eval.csv'

        ResNet_eval(resnet_model, save_path, config, device, attack, at_model, type_of_spec, df_eval=df_eval, epsilon=None)

    elif attack == 'Ensemble':
        epsilon_str = str(epsilon).replace('.', 'dot')

        if type_of_spec == 'mag':
            save_path = f'./eval/prob_ResNet_Ensemble_10_10_{epsilon_str}_mag.csv'
        elif type_of_spec == 'pow':
            save_path = f'./eval/prob_ResNet_Ensemble_10_10_{epsilon_str}.csv'

        ResNet_eval_ensemble(resnet_model, save_path, config, device, type_of_spec, epsilon, df_eval=None)

    else:
        print('todo')


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)

    config_path = 'config/residualnet_train_config.yaml'
    config_res = read_yaml(config_path)

    '''
    attack: 'FGSM' etc
    at_model: 'ResNet', 'SENet' (model used to perform the attack)
    epsilon: values like 1.0, 2.0....
    '''
    type_of_spec = 'pow'

    #init_eval(config_res, type_of_spec=type_of_spec, attack=None, at_model=None, epsilon=None)
    init_eval(config_res, type_of_spec=type_of_spec, attack='Ensemble', at_model=None, epsilon=3.0)
    #init_eval(config_res, type_of_spec=type_of_spec, attack='FGSM', at_model='SENet', epsilon=3.0)