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
from sklearn import model_selection
import os


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
            fname_list = []
            score_list = []
            feat_batch = feat_batch.to(torch.float32).to(device)
            score = model(feat_batch)
            score = (score[:, 0] - score[:, 1]).data.cpu().numpy().ravel()
            fname_list.extend(utt_id)
            score_list.extend(score.tolist())

            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list, score_list):
                    fh.write('{} {}\n'.format(f, cm))
            fh.close()
        print('Scores saved to {}'.format(save_path))


def init_eval(config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df_eval = pd.read_csv(config["df_eval_path"])

    if config['features'] == 'spec':
        resnet_spec_model = SpectrogramModel().to(device)
        resnet_spec_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device))

        save_path = './eval/scores_resnet_spec_eval.csv'
        resnet_eval(resnet_spec_model, df_eval, save_path, config, device)

    elif config['features'] == 'mfcc':
        resnet_mfcc_model = MFCCModel().to(device)
        resnet_mfcc_model.load_state_dict(torch.load(config['model_path_mfcc'], map_location=device))

        save_path = './eval/scores_resnet_mfcc_eval.csv'
        resnet_eval(resnet_mfcc_model, df_eval, save_path, config, device)


if __name__ == '__main__':

    seed_everything(1234)
    set_gpu(-1)

    config_path = 'config/residualnet_train_config.yaml'
    config_res = read_yaml(config_path)

    init_eval(config_res)

