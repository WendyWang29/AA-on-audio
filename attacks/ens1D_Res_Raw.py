from src.rawnet_utils import LoadAttackData_RawNet
from src.utils import *
import os
import gc
from src.ResNet1D.resnet1d_model import SpectrogramModel1D
from src.rawnet2_model import RawNet

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
import sys
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio

def Ens1D_ResRaw(config_res,
                 model_res, model_raw,
                 epsilon,
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_res, q_raw):

    epsilon_str = str(epsilon).replace('.', 'dot')
    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    audio_folder = f'Ens1D_ResRaw_{model_version}_{dataset}_{type_of_spec}_{q_res}_{q_raw}_{epsilon_str}'
    audio_folder = os.path.join(current_dir,
                                f'Ens1D_ResRaw_{model_version}_{type_of_spec}', audio_folder)

    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}...\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_RawNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     config=config_raw)
    data_loader = DataLoader(feat_set,
                             batch_size=config_raw['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set
    L = nn.NLLLoss()

    # parameters
    n_iters = 50
    alpha = epsilon / n_iters
    print(f'Using n_iters={n_iters} and alpha={alpha}\n')

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The Ensemble attack on ResNet and RawNet2 starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index, max_abs, mean in tqdm(data_loader, total=len(data_loader)):
        max_abs = max_abs.numpy()
        mean = mean.numpy()

        batch_x = batch_x.to(device)
        batch_z = batch_x.clone().to(device)
        batch_y = batch_y.to(device)

        batch_x.requires_grad = True
        batch_z.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iteration', leave=False) as pbar:
            for i in range(n_iters):

                out_res = model_res(batch_x)
                loss_res = L(out_res, batch_y)
                model_res.zero_grad()
                loss_res.backward()
                grad_res = batch_x.grad.data

                pert_batch = batch_x + alpha * grad_res.sign()

                # effect on ResNet
                predicted_labels_res = torch.argmax(model_res(pert_batch), dim=1)
                wrong_pred_res = (predicted_labels_res != batch_y)
                effect_res = wrong_pred_res.float().mean()*100

                # effect on RawNet
                predicted_labels_raw = torch.argmax(model_raw(pert_batch), dim=1)
                wrong_pred_raw = (predicted_labels_raw != batch_y)
                effect_raw = wrong_pred_raw.float().mean()*100

                pbar.set_description(
                    f'BIM iter {i + 1}/{n_iters} | eff. Res: {effect_res:.2f}% | eff. Raw: {effect_raw:.2f}%')

                batch_x = batch_z = pert_batch.detach().clone()
                batch_x.requires_grad = True
                batch_z.requires_grad = True

                pbar.update(1)

        pbar.n = min(pbar.total, i + 1)  # Update to the final iteration count
        pbar.refresh()
        batch_x = batch_x.squeeze(0).detach().cpu().numpy()

        print('ohoh')



if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config files
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path_res = os.path.join(script_dir, '../config/resnet1d.yaml')
    config_res = read_yaml(config_path_res)
    config_path_raw = os.path.join(script_dir, '../config/rawnet2.yaml')
    config_raw = read_yaml(config_path_raw)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'Ens1D_ResLCNN'  # 'FGSM' or 'BIM'
    dataset = 'whole'  # '3s' or 'whole'
    epsilon = 0.01
    model_version = 'v0'  # or 'old'
    type_of_spec = 'pow'  # 'pow' or 'mag'
    q_res = 5
    q_raw = 90
    '''
    #######################################
    '''

    # datset selection
    df_eval = pd.read_csv(os.path.join(script_dir, '..', config_raw['df_eval_path']))

    # load models
    model_res = SpectrogramModel1D().to(device)
    model_cls = RawNet(config_raw['model'], device)
    model_raw = model_cls.to(device)

    model_res.load_state_dict(
        torch.load(os.path.join(script_dir, '..', config_res['model_path_spec_pow_v0']), map_location=device))
    model_raw.load_state_dict(
        torch.load(os.path.join(script_dir, '..', config_raw['model_path_spec_pow_v0']), map_location=device))

    model_res.eval()
    model_raw.eval()

    Ens1D_ResRaw(config_res,
                 model_res, model_raw,
                 epsilon,
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_res, q_raw)