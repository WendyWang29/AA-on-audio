import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

logger = logging.getLogger("add_challenge")
logger.setLevel(logging.INFO)
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.SENet.SENet_model import se_resnet34_custom
from src.resnet_utils import LoadEvalData_ResNet
from src.utils import *
from tqdm import tqdm
import csv
import re
from sklearn import model_selection
import os

def SENet_eval(model, df_eval, save_path, config, device):
    file_eval = list(df_eval['path'])
    if os.path.exists(save_path):
        print(f'save_path exists, removing it to create a new one')
        os.system(f'rm {save_path}')
    '''
    using the same data loaders of ResNet
    '''
    feat_set = LoadEvalData_ResNet(list_IDs=file_eval, win_len=config_res['win_len'], config=config)
    feat_loader = DataLoader(feat_set, batch_size=config_res['eval_batch_size'], shuffle=False, num_workers=15)

    model.eval()

    with torch.no_grad():

        for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
            # fname_list = []
            # score_list = []
            feat_batch = feat_batch.to(torch.float32).to(device)
            score = model(feat_batch.unsqueeze(dim=1))
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


def init_eval(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    SENet_model = se_resnet34_custom(num_classes=2).to(device)

    print('Model state dictionary parameters:\n')
    for key, value in SENet_model.state_dict().items():
        print(f'{key}: {value.shape}')

    checkpoint = torch.load(config['model_path_spec'], map_location=device)
    print('\nCheckpoint state dict:\n')
    for key, value in checkpoint.items():
        print(f'{key}: {value.shape}')

    SENet_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device), strict=False)
    df_eval = pd.read_csv(config['df_eval_path'])
    save_path = './eval/prob_SENet_spec_eval.csv'

    SENet_eval(SENet_model, df_eval, save_path, config, device)

if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)

    config_path = 'config/SENet.yaml'
    config_res = read_yaml(config_path)

    init_eval(config_res)