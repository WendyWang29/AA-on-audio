import logging

from src.SENet.SENet_model import se_resnet34_custom

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


def init_eval(model, model_version, type_of_spec, dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    feature = 'audio'

    if model == 'ResNet':
        save_path = os.path.join(script_dir,
                                 'eval',
                                 f'probs_ResNet_{model_version}_clean_{dataset}_{type_of_spec}_{feature}.csv')
        resnet_model = SpectrogramModel().to(device)
        config_path = os.path.join(script_dir, 'config/residualnet_train_config.yaml')
        config = read_yaml(config_path)
        if type_of_spec == 'mag':
            if model_version == 'v0':
                resnet_model.load_state_dict(
                    torch.load(os.path.join(script_dir, config['model_path_spec_mag']), map_location=device),
                    strict=False)
        elif type_of_spec == 'pow':
            if model_version == 'v0':
                resnet_model.load_state_dict(
                    torch.load(os.path.join(script_dir, config['model_path_spec_pow_v0']), map_location=device),
                    strict=False)
        else:
            sys.exit('Wrong type of spectrogram mode: should be pow or mag')
    elif model == 'SENet':
        save_path = os.path.join(script_dir,
                                 'eval',
                                 f'probs_SENet_{model_version}_clean_{dataset}_{type_of_spec}_{feature}.csv')
        senet_model = se_resnet34_custom(num_classes=2).to(device)
        config_path = os.path.join(script_dir, 'config/SENet.yaml')
        config = read_yaml(config_path)
        if type_of_spec == 'mag':
            if model_version == 'v0':
                senet_model.load_state_dict(
                    torch.load(os.path.join(script_dir, config['model_path_spec_mag_v0']), map_location=device),
                    strict=False)
        elif type_of_spec == 'pow':
            if model_version == 'v0':
                senet_model.load_state_dict(
                    torch.load(os.path.join(script_dir, config['model_path_spec_pow_v0']), map_location=device),
                    strict=False)
        else:
            sys.exit('Wrong type of spectrogram mode: should be pow or mag')
    else:
        sys.exit('Wrong, TODO')

    if dataset == 'whole' and feature == 'audio':
        feat_directory = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/'
        feat_files = [f for f in os.listdir(feat_directory) if f.endswith('.flac')]
        csv_location = os.path.join(script_dir, 'eval',
                                    f'list_{feature}_{model}_{model_version}_{dataset}_{type_of_spec}')
    elif dataset == 'whole' and feature == 'mag_spec':
        sys.exit('TODO')
    elif dataset == '3s':
        sys.exit('TODO')

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
        sys.exit('TODO for spec')
        feat_set = LoadEvalData_ResNet_SPEC(list_IDs=file_eval, win_len=config_res['win_len'], config=config,
                                            type_of_spec=type_of_spec)
    elif feature == 'audio':
        feat_set = LoadEvalData_ResNet(list_IDs=file_eval, win_len=config_res['win_len'], config=config,
                                            type_of_spec=type_of_spec)
    else:
        sys.exit('Wrong type of feature, should be spec or audio')

    feat_loader = DataLoader(feat_set, batch_size=config_res['eval_batch_size'], shuffle=False, num_workers=15)

    if model == 'ResNet':
        resnet_model.eval()

        with torch.no_grad():

            for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
                # fname_list = []
                # score_list = []
                feat_batch = feat_batch.to(torch.float32).to(device)
                score = resnet_model(feat_batch)
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
    elif model == 'SENet':
        senet_model.eval()

        with torch.no_grad():

            for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
                # fname_list = []
                # score_list = []
                feat_batch = feat_batch.to(torch.float32).to(device)
                score = senet_model(feat_batch.unsqueeze(dim=1))
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







if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, 'config/residualnet_train_config.yaml')
    config_res = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    model = 'ResNet'
    model_version = 'v0'
    type_of_spec = 'pow'   # 'mag', 'pow'
    dataset = 'whole'   # '3s', 'whole'
    '''
    #######################################
    '''

    init_eval(model, model_version, type_of_spec, dataset)