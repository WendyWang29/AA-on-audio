from torchaudio.transforms import Spectrogram

from src.rawnet2_model import RawNet
from src.rawnet_utils import LoadAttackData_RawNet
from src.utils import *
import os
import gc

from src.SENet.senet1d_model import se_resnet341d_custom

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
import sys
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio



def Ens1D_RawSEN(config_sen,
                 model_raw,
                 model_sen,
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_raw, q_sen, eps_raw, eps_sen):

    torch.backends.cudnn.enabled = False
    epsilon_str_raw = str(eps_raw).replace('.', 'dot')
    epsilon_str_sen = str(eps_sen).replace('.', 'dot')

    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    audio_folder = f'Ens1D_RawSEN_{model_version}_{dataset}_{type_of_spec}_{q_raw}_{q_sen}_{epsilon_str_raw}_{epsilon_str_sen}'
    audio_folder = os.path.join(current_dir,
                                f'Ens1D_RawSEN_{model_version}_{type_of_spec}', audio_folder)

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
    alpha_raw = eps_raw / n_iters
    alpha_sen = eps_sen / n_iters
    print(f'Using n_iters={n_iters}')


    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The Ensemble attack on RawNet2 and SENet starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index, max_abs, mean in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()
        max_abs = max_abs.numpy()
        mean = mean.numpy()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_x.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iteration', leave=False) as pbar:
            for i in range(n_iters):

                out_sen = model_sen(batch_x)
                loss_sen = L(out_sen, batch_y)
                model_sen.zero_grad()
                loss_sen.backward(retain_graph=True)
                grad_sen = batch_x.grad.data.clone()
                batch_x.grad.zero_()

                out_raw = model_raw(batch_x)
                loss_raw = L(out_raw, batch_y)
                model_raw.zero_grad()
                loss_raw.backward(retain_graph=True)
                grad_raw = batch_x.grad.data.clone()
                batch_x.grad.zero_()

                # grad_raw_mean = torch.mean(grad_raw)
                # grad_raw_std = torch.std(grad_raw)
                # grad_sen_mean = torch.mean(grad_sen)
                # grad_sen_std = torch.std(grad_sen)

                '''
                compute thresholds
                '''
                thresh_raw = torch.quantile(torch.abs(grad_raw), q_raw / 100)
                thresh_sen = torch.quantile(torch.abs(grad_sen), q_sen / 100)

                mask_raw = torch.abs(grad_raw) > thresh_raw
                mask_sen = torch.abs(grad_sen) > thresh_sen

                '''
                create new grad
                '''
                overlap_mask = mask_raw & mask_sen  # intersection

                # Create pert_batch in-place
                pert_batch = batch_x.clone()
                pert_batch[mask_raw & ~overlap_mask] += alpha_raw * grad_raw[mask_raw & ~overlap_mask].sign()
                pert_batch[mask_sen & ~overlap_mask] += alpha_sen * grad_sen[mask_sen & ~overlap_mask].sign()

                '''
                Handle overlapping values: half from grad_raw, half from grad_raw
                '''

                # For the overlap region, randomly mix half of grad_sen and grad_raw
                overlap_indices = overlap_mask.nonzero(as_tuple=True)

                if overlap_indices[0].numel() > 0:
                    # Randomly select half the positions for grad_raw and the other half for grad_sen
                    random_mask_raw = torch.rand(overlap_indices[0].numel()).to(
                        grad_raw.device) < 0.5  # 50% mask for grad_raw
                    random_mask_sen = ~random_mask_raw  # The remaining 50% mask for grad_sen

                    # Apply grad_raw to the positions selected by random_mask_raw
                    pert_batch[overlap_indices[0][random_mask_raw], overlap_indices[1][random_mask_raw]] += \
                        alpha_raw * grad_raw[
                            overlap_indices[0][random_mask_raw], overlap_indices[1][random_mask_raw]].sign()

                    # Apply grad_raw to the positions selected by random_mask_raw
                    pert_batch[overlap_indices[0][random_mask_sen], overlap_indices[1][random_mask_sen]] += \
                        alpha_sen * grad_sen[
                            overlap_indices[0][random_mask_sen], overlap_indices[1][random_mask_sen]].sign()

                # effect on RawNet2
                predicted_labels_raw = torch.argmax(model_raw(pert_batch), dim=1)
                wrong_pred_raw = (predicted_labels_raw != batch_y)
                effect_raw = wrong_pred_raw.float().mean() * 100

                # effect on SENet
                predicted_labels_sen = torch.argmax(model_sen(pert_batch), dim=1)
                wrong_pred_sen = (predicted_labels_sen != batch_y)
                effect_sen = wrong_pred_sen.float().mean() * 100

                pbar.set_description(f'BIM iter {i + 1}/{n_iters} | eff. SEN: {effect_sen:.2f}% | eff. Raw: {effect_raw:.2f}%')

                batch_x = pert_batch.detach().clone()
                batch_x.requires_grad = True
                del grad_raw, grad_sen, pert_batch, predicted_labels_raw, predicted_labels_sen, wrong_pred_sen, wrong_pred_raw
                torch.cuda.empty_cache()
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

            # cut the audio to original audio length
            sliced_audio = audio[:audio_len[m]]

            save_perturbed_audio(file=file_eval[index[m]],
                                 folder=audio_folder,
                                 audio=sliced_audio,
                                 sr=16000,
                                 attack='Ens1D',
                                 epsilon=None,
                                 model=f'RawSEN_{q_raw}_{q_sen}',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)
        del batch_x, pert_batch

        time_taken = time.time() - start_time
        tqdm.write(
            f'Time taken: {time_taken:.3f} | SEN effect. {effect_sen:.2f}% | Raw eff. {effect_raw:.2f}% ')
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config files
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path_sen = os.path.join(script_dir, '../config/senet1d.yaml')
    config_sen = read_yaml(config_path_sen)
    config_path_raw = os.path.join(script_dir, '../config/rawnet2.yaml')
    config_raw = read_yaml(config_path_raw)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'Ens1D_RawSEN'   #'FGSM' or 'BIM'
    dataset = 'whole'  # '3s' or 'whole'
    model_version = 'v0' # or 'old'
    type_of_spec = 'pow'   # 'pow' or 'mag'
    q_raw = 50
    q_sen = 40
    eps_raw = 0.008
    eps_sen = 0.004
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
    model_cls = RawNet(config_raw['model'], device)
    model_raw = model_cls.to(device)

    model_sen.load_state_dict(torch.load(os.path.join(script_dir, '..', config_sen['model_path_spec_pow_v0']), map_location=device))
    model_raw.load_state_dict(torch.load(os.path.join(script_dir, '..', config_raw['model_path_spec_pow_v0']), map_location=device))

    model_sen.eval()
    model_raw.eval()

    Ens1D_RawSEN(config_sen,
                 model_raw,
                 model_sen,
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_raw, q_sen, eps_raw, eps_sen)