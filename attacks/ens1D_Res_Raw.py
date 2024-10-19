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
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_res, q_raw, eps_res, eps_raw):

    torch.backends.cudnn.enabled = False
    epsilon_str_res = str(eps_res).replace('.', 'dot')
    epsilon_str_raw = str(eps_raw).replace('.', 'dot')

    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    audio_folder = f'Ens1D_ResRaw_{model_version}_{dataset}_{type_of_spec}_{q_res}_{q_raw}_{epsilon_str_res}_{epsilon_str_raw}'
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
    alpha_res = eps_res / n_iters
    alpha_raw = eps_raw / n_iters
    print(f'Using n_iters={n_iters}')


    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The Ensemble attack on ResNet and RawNet2 starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index, max_abs, mean in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()
        max_abs = max_abs.numpy()
        mean = mean.numpy()

        batch_z = batch_x.clone().to(device)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_x.requires_grad = True
        batch_z.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iteration', leave=False) as pbar:
            for i in range(n_iters):

                '''
                compute gradients
                '''
                out_res = model_res(batch_x)
                loss_res = L(out_res, batch_y)
                model_res.zero_grad()
                loss_res.backward()
                grad_res = batch_x.grad.data

                out_raw = model_raw(batch_z)
                loss_raw = L(out_raw, batch_y)
                model_raw.zero_grad()
                loss_raw.backward()
                grad_raw = batch_z.grad.data

                # grad_res_mean = torch.mean(grad_res)
                # grad_res_std = torch.std(grad_res)
                # grad_raw_mean = torch.mean(grad_raw)
                # grad_raw_std = torch.std(grad_raw)

                '''
                compute percentile thresholds
                '''
                thresh_res = torch.quantile(torch.abs(grad_res), q_res/100)
                thresh_raw = torch.quantile(torch.abs(grad_raw), q_raw/100)

                mask_res = torch.abs(grad_res) > thresh_res
                mask_raw = torch.abs(grad_raw) > thresh_raw

                '''
                create new gradient
                '''
                overlap_mask = mask_res & mask_raw  # intersection

                # Take the sign of grad_res and grad_raw for non-overlapping and apply corresponding alpha
                pert_batch = batch_x.clone()  # Clone batch_x to pert_batch

                # Apply alpha_res to grad_res for non-overlapping areas
                pert_batch[mask_res & ~overlap_mask] += alpha_res * grad_res[mask_res & ~overlap_mask].sign()

                # Apply alpha_raw to grad_raw for non-overlapping areas
                pert_batch[mask_raw & ~overlap_mask] += alpha_raw * grad_raw[mask_raw & ~overlap_mask].sign()

                '''
                Handle overlapping values: half from grad_res, half from grad_raw
                '''
                # For the overlap region, randomly mix half of grad_res and grad_raw
                overlap_indices = overlap_mask.nonzero(as_tuple=True)

                if overlap_indices[0].numel() > 0:
                    # Randomly select half the positions for grad_res and the other half for grad_raw
                    random_mask_res = torch.rand(overlap_indices[0].numel()).to(
                        grad_res.device) < 0.5  # 50% mask for grad_res
                    random_mask_raw = ~random_mask_res  # The remaining 50% mask for grad_raw

                    # Apply grad_res to the positions selected by random_mask_res
                    pert_batch[overlap_indices[0][random_mask_res], overlap_indices[1][random_mask_res]] += \
                        alpha_res * grad_res[
                            overlap_indices[0][random_mask_res], overlap_indices[1][random_mask_res]].sign()

                    # Apply grad_raw to the positions selected by random_mask_raw
                    pert_batch[overlap_indices[0][random_mask_raw], overlap_indices[1][random_mask_raw]] += \
                        alpha_raw * grad_raw[
                            overlap_indices[0][random_mask_raw], overlap_indices[1][random_mask_raw]].sign()

                # effect on ResNet
                predicted_labels_res = torch.argmax(model_res(pert_batch), dim=1)
                wrong_pred_res = (predicted_labels_res != batch_y)
                effect_res = wrong_pred_res.float().mean()*100

                # effect on RawNet
                predicted_labels_raw = torch.argmax(model_raw(pert_batch), dim=1)
                wrong_pred_raw = (predicted_labels_raw != batch_y)
                effect_raw = wrong_pred_raw.float().mean()*100

                pbar.set_description(
                    f'BIM iter {i + 1}/{n_iters} | eff. Res: {effect_res:.2f}% | eff. Raw: {effect_raw:.2f}% ')

                batch_x = pert_batch.detach().clone()
                batch_z = pert_batch.detach().clone()
                batch_x.requires_grad = True
                batch_z.requires_grad = True

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
                                 attack='Ens1D',
                                 epsilon=None,
                                 model=f'ResRaw_{q_res}_{q_raw}',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)
        del batch_x

        time_taken = time.time() - start_time
        tqdm.write(
            f'Time taken: {time_taken:.3f} | Res eff. {effect_res:.2f}% | Raw effect. {effect_raw:.2f}%')
        gc.collect()



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
    attack = 'Ens1D_ResRaw'  # 'FGSM' or 'BIM'
    dataset = 'whole'  # '3s' or 'whole'
    model_version = 'v0'  # or 'old'
    type_of_spec = 'pow'  # 'pow' or 'mag'
    q_res = 30
    q_raw = 50
    eps_res = 0.008
    eps_raw = 0.02
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
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_res, q_raw, eps_res, eps_raw)