from src.utils import *
import os
import gc
from src.resnet_model import SpectrogramModel
from src.resnet_utils import LoadAttackData_ResNet

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
import sys
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio, save_perturbed_spec








def compute_specs(df_eval, type_of_spec):


    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec_folder = os.path.join(current_dir, f'whole_dataset_pow_specs')

    os.makedirs(spec_folder, exist_ok=True)
    print(f'Saving the power specs in {spec_folder}...\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_ResNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     win_len=config['win_len'],
                                     config=config,
                                     type_of_spec=type_of_spec)
    data_loader = DataLoader(feat_set,
                             batch_size=config['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set


    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The pow spec computation starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸')

    effect = []  # for storing the unbalanced effectiveness on Resnet

    win_length = 2048
    n_fft = 2048
    hop_length = 512
    window = 'hann'

    # ########## ATTACK ##########
    for batch_x, batch_y, phase, audio_len, index in tqdm(data_loader, total=len(data_loader)):

        for i in range(batch_x.shape[0]):

            # save the spec as a (1025,93) spec for all specs
            spec = batch_x[i]
            save_perturbed_spec(file=file_eval[index[i]],
                                folder=spec_folder,
                                spec=spec,
                                epsilon=None,
                                attack=None,
                                model=None,
                                model_version='v0',
                                type_of_spec=type_of_spec)


        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, '../config/residualnet_train_config.yaml')
    config = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    dataset = 'whole'  # '3s' or 'whole'
    type_of_spec = 'pow'   # 'pow' or 'mag'
    '''
    #######################################
    '''

    # load the dataset to work on
    if dataset == 'whole':
        # load the entire ASVSpoof2019 eval dataset
        df_eval = pd.read_csv(os.path.join(script_dir, '..', config['df_eval_path']))
    elif dataset == '3s':
        # load the reduced dataset containing only audio >3s
        df_eval = pd.read_csv(os.path.join(script_dir, '..', config['df_eval_path_3s']))
    else:
        sys.exit(f'You need to define the dataset to work on, {dataset} is not valid')


    compute_specs(df_eval, type_of_spec)
