import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

logger = logging.getLogger("add_challenge")
logger.setLevel(logging.INFO)
import torch
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.utils import *
from src.rawnet2_model import RawNet
from eval_resnet import create_csv
import csv
from tqdm import tqdm
from src.rawnet_utils import LoadEvalData_RawNet

def rawnet_eval(model, df_eval, save_path, config, device):
    file_eval = list(df_eval['path'])

    if os.path.exists(save_path):
        print(f'save_path exists, removing it to create a new one')
        os.system(f'rm {save_path}')

    feat_set = LoadEvalData_RawNet(list_IDs=file_eval, win_len=config['win_len'], config=config)
    feat_loader = DataLoader(feat_set, batch_size=config['eval_batch_size'], shuffle=False, num_workers=15)
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
        print('Scores saved to {}'.format(save_path))


def init_eval(config, attack=None, epsilon=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rawnet_model = RawNet(config['model'], device)
    rawnet_model = rawnet_model.to(device)
    rawnet_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device))

    if attack is not None:
        # df_eval will be a list of perturbed flac files
        path_to_csv = create_csv(attack, epsilon)
        df_eval = pd.read_csv(path_to_csv)
        epsilon = str(epsilon).replace('.', 'dot')
        save_path = f'./eval/prob_rawnet_eval_{attack}_{epsilon}.csv'
        rawnet_eval(rawnet_model, df_eval, save_path, config, device)
    else:
        # no attack performed
        df_eval = pd.read_csv(config["df_eval_path"])
        save_path = './eval/prob_resnet_eval.csv'
        rawnet_eval(rawnet_model, df_eval, save_path, config, device)



if __name__ == '__main__':

    seed_everything(1234)
    set_gpu(-1)

    config_path = 'config/rawnet2.yaml'
    config = read_yaml(config_path)

    init_eval(config, attack='FGSM', epsilon=0.2)