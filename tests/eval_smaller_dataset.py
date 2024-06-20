from src.utils import *
from src.rawnet2_model import RawNet
from src.resnet_model import SpectrogramModel
from eval_resnet import create_csv
from src.resnet_utils import LoadEvalData_ResNet
from src.rawnet_utils import LoadEvalData_RawNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os




def eval_smaller_dataset(model, model_, df_eval, save_path, config, device):

    # create list of the perturbed flac files paths
    file_eval = list(df_eval['path'])

    if os.path.exists(save_path):
        print(f'save_path exists, removing it to create a new one')
        os.system(f'rm {save_path}')

    # create data loader based on type of model
    if model == 'ResNet':
        feat_set = LoadEvalData_ResNet(list_IDs=file_eval, win_len=config['win_len'], config=config)
        feat_loader = DataLoader(feat_set, batch_size=config['eval_batch_size'], shuffle=False, num_workers=15)
    elif model == 'RawNet2':
        feat_set = LoadEvalData_RawNet(list_IDs=file_eval, config=config)
        feat_loader = DataLoader(feat_set, batch_size=config['eval_batch_size'], shuffle=False, num_workers=15)

    model_.eval()  #could be ResNet or RawNet2

    with torch.no_grad():
        for feat_batch, utt_id in tqdm(feat_loader, total=len(feat_loader)):
            feat_batch = feat_batch.to(torch.float32).to(device)
            score = model_(feat_batch)
            probabilities = torch.exp(score)
            probabilities = probabilities.detach().cpu().numpy()

            with open(save_path, mode='a+', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(['Filename', 'prob. class 0', 'prob. class 1'])
                for i in range(len(utt_id)):
                    row = [utt_id[i], probabilities[i, 0], probabilities[i, 1]]
                    writer.writerow(row)



def init_eval(config, model, attack, epsilon):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create a csv file to store the probabilities
    path_to_csv = create_csv(attack, epsilon)
    df_eval = pd.read_csv(path_to_csv)
    epsilon = str(epsilon).replace('.', 'dot')

    if model == 'RawNet2':
        rawnet_model = RawNet(config['model'], device)
        rawnet_model = rawnet_model.to(device)
        rawnet_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device))
        model_ = rawnet_model
        save_path = f'../eval/prob_rawnet_eval_{attack}_{epsilon}.csv'
    elif model == 'ResNet':
        resnet_spec_model = SpectrogramModel().to(device)
        resnet_spec_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device))
        model_ = resnet_spec_model
        save_path = f'../eval/prob_resnet_eval_{attack}_{epsilon}.csv'

    # following works both with ResNet and RawNet2
    eval_smaller_dataset(model, model_, df_eval, save_path, config, device)










if __name__ == '__main__':
    '''
    This script is meant to work on smaller datasets wrt the complete one
    It outputs a csv file with the probabilities for class 0 and class 1
    for each flac file in the perturbed smaller dataset
    '''
    seed_everything(1234)
    set_gpu(-1)

    ######### things to set #########
    #model = 'RawNet2'
    model = 'ResNet'
    attack = 'SSA'
    #attack = 'FGSM'
    epsilon = 5.0
    #################################

    if model == 'RawNet2':
        config_path = '../config/rawnet2.yaml'
    elif model == 'ResNet':
        config_path = '../config/residualnet_train_config.yaml'
    else:
        print('Model must be "RawNet2" or "ResNet"')

    config = read_yaml(config_path)
    init_eval(config, model, attack, epsilon)