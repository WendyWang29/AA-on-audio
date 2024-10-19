from torchaudio.transforms import Spectrogram

from src.rawnet_utils import LoadAttackData_RawNet
from src.utils import *
import os
import gc
from src.ResNet1D.resnet1d_model import SpectrogramModel1D
from src.SENet.senet1d_model import se_resnet341d_custom

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
import sys
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio



def Ens1D_ResSEN(config_sen,
                 model_res,
                 model_sen,
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_res, q_sen, eps_res, eps_sen):

    torch.backends.cudnn.enabled = False
    epsilon_str_res = str(eps_res).replace('.', 'dot')
    epsilon_str_sen = str(eps_sen).replace('.', 'dot')

    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    audio_folder = f'Ens1D_ResSEN_{model_version}_{dataset}_{type_of_spec}_{q_res}_{q_sen}_{epsilon_str_res}_{epsilon_str_sen}'
    audio_folder = os.path.join(current_dir,
                                f'Ens1D_ResSEN_{model_version}_{type_of_spec}', audio_folder)

    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}...\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_RawNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     config=config_sen)
    data_loader = DataLoader(feat_set,
                             batch_size=config_sen['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set
    L = nn.NLLLoss()

    # parameters
    n_iters = 50
    alpha_res = eps_res / n_iters
    alpha_sen = eps_sen / n_iters
    print(f'Using n_iters={n_iters}')


    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The Ensemble attack on ResNet and SENet starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index, max_abs, mean in tqdm(data_loader, total=len(data_loader)):

        max_abs = max_abs.numpy()
        mean = mean.numpy()

        batch_z = batch_x.clone().to(device)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_x.requires_grad = True
        batch_z.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iteration', leave=False) as pbar:
            for i in range(n_iters):

                out_sen = model_sen(batch_x)
                loss_sen = L(out_sen, batch_y)
                model_sen.zero_grad()
                loss_sen.backward()
                grad_sen = batch_x.grad.data

                out_res = model_res(batch_z)
                loss_res = L(out_res, batch_y)
                model_res.zero_grad()
                loss_res.backward()
                grad_res = batch_z.grad.data

                # grad_res_mean = torch.mean(grad_res)
                # grad_res_std = torch.std(grad_res)
                # grad_sen_mean = torch.mean(grad_sen)
                # grad_sen_std = torch.std(grad_sen)

                '''
                compute thresholds
                '''
                thresh_res = torch.quantile(torch.abs(grad_res), q_res / 100)
                thresh_sen = torch.quantile(torch.abs(grad_sen), q_sen / 100)

                mask_res = torch.abs(grad_res) > thresh_res
                mask_sen = torch.abs(grad_sen) > thresh_sen

                '''
                create new grad
                '''
                overlap_mask = mask_res & mask_sen  # intersection

                # Take the sign of grad_res and grad_raw for non-overlapping and apply corresponding alpha
                pert_batch = batch_x.clone()  # Clone batch_x to pert_batch

                # Apply alpha_res to grad_res for non-overlapping areas
                pert_batch[mask_res & ~overlap_mask] += alpha_res * grad_res[mask_res & ~overlap_mask].sign()

                # Apply alpha_sen to grad_raw for non-overlapping areas
                pert_batch[mask_sen & ~overlap_mask] += alpha_sen * grad_sen[mask_sen & ~overlap_mask].sign()

                '''
                Handle overlapping values: half from grad_res, half from grad_raw
                '''
                if overlap_mask.sum() > 0:
                    # Get overlapping values for grad_res and grad_raw
                    overlap_values_res = grad_res[overlap_mask]
                    overlap_values_sen = grad_sen[overlap_mask]

                    # Sort the absolute values and get the indices for sorting
                    sorted_res_idx = torch.argsort(torch.abs(overlap_values_res), descending=True)
                    sorted_sen_idx = torch.argsort(torch.abs(overlap_values_sen), descending=True)

                    # Split indices in half: top half from grad_res, top half from grad_raw
                    n_overlap = overlap_values_res.numel()
                    half_size = n_overlap // 2

                    # Take half of the top values from grad_res
                    top_res_idx = sorted_res_idx[:half_size]

                    # Take half of the top values from grad_raw
                    top_sen_idx = sorted_sen_idx[:half_size]

                    # Initialize new gradient for the overlapping mask
                    new_overlap_grad = torch.zeros_like(overlap_values_res)

                    # Assign top values from grad_res
                    new_overlap_grad[top_res_idx] = alpha_res * overlap_values_res[top_res_idx].sign()

                    # Assign top values from grad_raw
                    new_overlap_grad[top_sen_idx] = alpha_sen * overlap_values_sen[top_sen_idx].sign()

                    # Update pert_batch with the new values for overlapping region
                    pert_batch[overlap_mask] += new_overlap_grad

                out_pert_sen = model_sen(pert_batch)
                predicted_labels_sen = torch.argmax(out_pert_sen, dim=1)
                wrong_predictions_sen = (predicted_labels_sen != batch_y)
                effectiveness_sen = wrong_predictions_sen.float().mean()
                effectiveness_percentage_sen = effectiveness_sen * 100

                out_pert_res = model_res(pert_batch)
                predicted_labels_res = torch.argmax(out_pert_res, dim=1)
                wrong_predictions_res = (predicted_labels_res != batch_y)
                effectiveness_res = wrong_predictions_res.float().mean()
                effectiveness_percentage_res = effectiveness_res * 100

                pbar.set_description(f'BIM iter {i + 1}/{n_iters} | eff. SEN: {effectiveness_percentage_sen:.2f}% | eff. Res: {effectiveness_percentage_res:.2f}%')

                batch_x = batch_z = pert_batch.detach().clone()
                batch_x.requires_grad = True
                batch_z.requires_grad = True

                del grad_res, grad_sen

                pbar.update(1)

        pbar.n = min(pbar.total, i + 1)  # Update to the final iteration count
        pbar.refresh()
        batch_x = batch_x.squeeze(0).detach().cpu().numpy()

        """
        Save audio
        """
        for m in range(batch_x.shape[0]):

            # get the single perturbed audio
            audio = batch_x[m]
            clean_max_abs = max_abs[m]
            clean_mean = mean[m]

            # normalize to have the same max abs value
            pert_max_abs = np.max(np.abs(audio))
            if pert_max_abs > 0:  # avoid division by 0
                audio = audio * (clean_max_abs / pert_max_abs)

            # same mean
            pert_mean = np.mean(audio)
            audio = audio + (clean_mean - pert_mean)

            save_perturbed_audio(file=file_eval[index[m]],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 attack=attack,
                                 epsilon=None,
                                 model=f'Ens1D_ResSEN_{q_res}_{q_sen}',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)
        del batch_x


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config files
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path_sen = os.path.join(script_dir, '../config/senet1d.yaml')
    config_sen = read_yaml(config_path_sen)
    config_path_res = os.path.join(script_dir, '../config/resnet1d.yaml')
    config_res = read_yaml(config_path_res)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'Ens1D_ResSEN'   #'FGSM' or 'BIM'
    dataset = 'whole'  # '3s' or 'whole'
    model_version = 'v0' # or 'old'
    type_of_spec = 'pow'   # 'pow' or 'mag'
    q_res = 40
    q_sen = 40
    eps_res = eps_sen = 0.001
    '''
    #######################################
    '''

    # load the dataset to work on
    if dataset == 'whole':
        # load the entire ASVSpoof2019 eval dataset
        df_eval = pd.read_csv(os.path.join(script_dir, '..', config_sen['df_eval_path']))
    elif dataset == '3s':
        # load the reduced dataset containing only audio >3s
        df_eval = pd.read_csv(os.path.join(script_dir, '..', config_sen['df_eval_path_3s']))
    else:
        sys.exit(f'You need to define the dataset to work on, {dataset} is not valid')

    # load the models
    model_sen = se_resnet341d_custom(num_classes=2).to(device)
    model_res = SpectrogramModel1D().to(device)

    model_sen.load_state_dict(torch.load(os.path.join(script_dir, '..', config_sen['model_path_spec_pow_v0']), map_location=device))
    model_res.load_state_dict(torch.load(os.path.join(script_dir, '..', config_res['model_path_spec_pow_v0']), map_location=device))

    model_sen.eval()
    model_res.eval()

    Ens1D_ResSEN(config_sen,
                 model_res,
                 model_sen,
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_res, q_sen, eps_res, eps_sen)