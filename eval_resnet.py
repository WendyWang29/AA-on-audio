import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

logger = logging.getLogger("add_challenge")
logger.setLevel(logging.INFO)
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.resnet_model import SpectrogramModel, MFCCModel
from src.resnet_utils import LoadEvalData_ResNet
from src.utils import *
from tqdm import tqdm
import csv
import re
from sklearn import model_selection
import os


def create_csv(attack, epsilon):

    epsilon_dot_notation = str(epsilon).replace('.', 'dot')
    flac_directory = os.path.join('attacks', f'{attack}_data', f'{attack}_dataset_{epsilon_dot_notation}')

    # specify full path in which csv file has to be saved
    csv_location = os.path.join('eval', f'flac_{attack}_{epsilon_dot_notation}_specs.csv')
    if os.path.exists(csv_location):
        os.remove(csv_location)
        print(f"Existing file '{csv_location}' has been removed.")

    # def extract_number_from_path(file_path):
    #     match = re.search(r'_(\d+)\.flac$', file_path)
    #     if match:
    #         return match.group(1)
    #     else:
    #         return None
    #
    # transformed_rows = []
    #
    # with open(df_eval_original, 'r') as file:
    #     csv_reader = csv.reader(file)
    #     for row in csv_reader:
    #         index = row[0]
    #         df_eval_original = row[1]
    #         number = extract_number_from_path(df_eval_original)
    #         if number is not None:
    #             new_path = f'attacks/FGSM_data/FGSM_dataset_0dot0/FGSM_LA_E_{number}_{epsilon_dot_notation}.flac'
    #             transformed_rows.append([new_path])
    #
    # with open(csv_location, 'w', newline='') as file:
    #     csv_writer = csv.writer(file)
    #     csv_writer.writerow(transformed_rows)
    #
    # return csv_location

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

def resnet_eval(model, df_eval, save_path, config, device):

    try:
        file_eval = list(df_eval['Path'])
    except:
        file_eval = list(df_eval['path'])

    if os.path.exists(save_path):
        print(f'save_path exists, removing it to create a new one')
        os.system(f'rm {save_path}')

    feat_set = LoadEvalData_ResNet(list_IDs=file_eval, win_len=config_res['win_len'], config=config)
    feat_loader = DataLoader(feat_set, batch_size=config_res['eval_batch_size'], shuffle=False, num_workers=15)

    model.eval()

    with torch.no_grad():

        for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
            #fname_list = []
            #score_list = []
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



            #score = (score[:, 0] - score[:, 1]).data.cpu().numpy().ravel()
            # fname_list.extend(utt_id)
            # score_list.extend(score.tolist())
            #
            # with open(save_path, 'a+') as fh:
            #     for f, cm in zip(fname_list, score_list):
            #         fh.write('{} {} {}\n'.format(f, cm))
            # fh.close()
        print('Scores saved to {}'.format(save_path))


def init_eval(config, attack=None, epsilon=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config['features'] == 'spec':
        resnet_spec_model = SpectrogramModel().to(device)
        resnet_spec_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device))

        if attack is not None:
            # df_eval will be a list of perturbed flac files
            path_to_csv = create_csv(attack, epsilon)
            df_eval = pd.read_csv(path_to_csv)
            epsilon = str(epsilon).replace('.', 'dot')
            save_path = f'./eval/prob_resnet_spec_eval_{attack}_{epsilon}.csv'
            resnet_eval(resnet_spec_model, df_eval, save_path, config, device)
        else:
            # no attack performed
            df_eval = pd.read_csv(config["df_eval_path"])
            save_path = './eval/prob_resnet_spec_eval.csv'
            resnet_eval(resnet_spec_model, df_eval, save_path, config, device)

    elif config['features'] == 'mfcc':
        print('ACtually we are not working with MFCCs...')
        # resnet_mfcc_model = MFCCModel().to(device)
        # resnet_mfcc_model.load_state_dict(torch.load(config['model_path_mfcc'], map_location=device))
        #
        # save_path = './eval/scores_resnet_mfcc_eval.csv'
        # resnet_eval(resnet_mfcc_model, df_eval, save_path, config, device)


if __name__ == '__main__':

    seed_everything(1234)
    set_gpu(-1)

    config_path = 'config/residualnet_train_config.yaml'
    config_res = read_yaml(config_path)

    init_eval(config_res, attack='FGSM', epsilon=0.6)

