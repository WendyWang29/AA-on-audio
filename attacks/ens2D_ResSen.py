from src.SENet.SENet_model import se_resnet34_custom
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


def Ens2D_ResSEN(model_res, model_sen, df_eval, config_res, q_res, q_sen, epsilon, device, model_version, dataset):

    # create the folder for the perturbed dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    type_of_spec = 'pow'
    epsilon_str = str(epsilon).replace('.', 'dot')

    # audio and spec folder
    audio_folder = f'Ens2D_ResSEN_{model_version}_{dataset}_{type_of_spec}_{epsilon_str}'
    audio_folder = os.path.join(current_dir, f'Ens2D_ResSEN_{model_version}_{type_of_spec}', audio_folder)
    spec_folder = os.path.join(current_dir, f'Ens2D_ResSEN_{model_version}_{type_of_spec}', audio_folder, 'spec')

    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(spec_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_ResNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     win_len=config_res['win_len'],
                                     config=config_res,
                                     type_of_spec=type_of_spec)
    data_loader = DataLoader(feat_set,
                             batch_size=config_res['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set
    L = nn.NLLLoss()

    n_iters = 50  # max number of BIM iterations
    alpha = epsilon / n_iters  # perturbation to add at each iteration
    print(f'Using n_iters={n_iters} and alpha={alpha}\n')

    win_length = 2048
    n_fft = 2048
    hop_length = 512
    eps = 1e-20
    window = 'hann'

    w_res = 1
    w_sen = 0.01

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The ensemble attack 2D on ResNet and SENet starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, phase, audio_len, index in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        phase = phase.numpy()
        batch_x = batch_x.to(device)
        batch_z = batch_x.clone().to(device)
        batch_y = batch_y.to(device)

        batch_x.requires_grad = True
        batch_z.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iterations', leave=False) as pbar:
            for i in range(n_iters):

                """
                get the gradients
                """
                out_res = model_res(batch_x)
                loss_res = L(out_res, batch_y)
                model_res.zero_grad()
                loss_res.backward()
                grad_res = batch_x.grad.data

                out_sen = model_sen(batch_z.unsqueeze(dim=1))
                loss_sen = L(out_sen, batch_y)
                model_sen.zero_grad()
                loss_sen.backward()
                grad_sen = batch_z.grad.data

                """
                study the gradients
                """
                # grad_res_mean = torch.mean(grad_res)
                # grad_res_std = torch.std(grad_res)
                # grad_sen_mean = torch.mean(grad_sen)
                # grad_sen_std = torch.std(grad_sen)

                """
                compute thresholds
                """
                abs_grad_sen = torch.abs(grad_sen)
                abs_grad_res = torch.abs(grad_res)
                thresh_sen = torch.quantile(abs_grad_sen, q_sen / 100)
                thresh_res = torch.quantile(abs_grad_res, q_res / 100)

                """
                create the new grad
                """
                matrix = torch.full_like(grad_sen, float('inf'))

                # Masks for values that exceed the threshold
                mask_sen = abs_grad_sen > thresh_sen
                mask_res = abs_grad_res > thresh_res

                # Overlap where both grad_res and grad_sen exceed their thresholds
                overlap = mask_sen & mask_res

                # For overlapping values, compute the mean of grad_res and grad_sen
                #matrix[overlap] = (w_res *grad_res[overlap] + w_sen*grad_sen[overlap]) / 2
                matrix[overlap] = (w_res * grad_res[overlap])

                # For non-overlapping values, take grad_sen values where only grad_sen exceeds the threshold
                matrix[mask_sen & ~overlap] = w_sen*grad_sen[mask_sen & ~overlap]

                # For non-overlapping values, take grad_res values where only grad_res exceeds the threshold
                matrix[mask_res & ~overlap] = w_res*grad_res[mask_res & ~overlap]

                # replace inf with 0
                matrix[matrix == float('inf')] = 0
                grad = matrix

                """
                Perform the attack
                """
                pert_batch = batch_x + alpha * grad.sign()

                out_pert_res = model_res(pert_batch)
                predicted_labels_res = torch.argmax(out_pert_res, dim=1)
                wrong_predictions_res = (predicted_labels_res != batch_y)
                effectiveness_res = wrong_predictions_res.float().mean()
                effectiveness_percentage_res = effectiveness_res * 100

                out_pert_sen = model_sen(pert_batch.unsqueeze(dim=1))
                predicted_labels_sen = torch.argmax(out_pert_sen, dim=1)
                wrong_predictions_sen = (predicted_labels_sen != batch_y)
                effectiveness_sen = wrong_predictions_sen.float().mean()
                effectiveness_percentage_sen = effectiveness_sen * 100

                pbar.set_description(f'BIM iter {i + 1}/{n_iters} | eff. SEN: {effectiveness_percentage_sen:.2f}% | eff. Res: {effectiveness_percentage_res:.2f}%')

                batch_x = batch_z = pert_batch.detach().clone()
                batch_x.requires_grad = True
                batch_z.requires_grad = True

                pbar.update(1)

        pbar.n = min(pbar.total, i + 1)  # Update to the final iteration count
        pbar.refresh()
        batch_x = batch_x.squeeze(0).detach().cpu().numpy()

        """
        Save spec and audio
        """
        for m in range(batch_x.shape[0]):
            # save the spec as a (1025,93) spec for all specs
            spec = batch_x[m]
            save_perturbed_spec(file=file_eval[index[m]],
                                folder=spec_folder,
                                spec=spec,
                                epsilon=epsilon,
                                attack=attack,
                                model='ResNet2D',
                                model_version=model_version,
                                type_of_spec=type_of_spec)

            # spectrogram inversion
            if type_of_spec == 'mag':
                mag = spec
            elif type_of_spec == 'pow':
                linear = librosa.db_to_power(spec)
                mag = np.sqrt(linear)
            else:
                sys.exit('You shouldnt be here')

            phase_single_audio = phase[m]
            recon_audio = librosa.istft(mag * np.exp(1j * phase_single_audio),
                                        n_fft=n_fft,
                                        window=window,
                                        win_length=win_length,
                                        hop_length=hop_length,
                                        center=True)

            if type_of_spec == 'pow':
                recon_audio = librosa.util.normalize(recon_audio)
            else:
                pass

            # cut the audio to original audio length
            sliced_audio = recon_audio[:audio_len[m]]

            save_perturbed_audio(file=file_eval[index[m]],
                                 folder=audio_folder,
                                 audio=sliced_audio,
                                 sr=16000,
                                 attack=attack,
                                 epsilon=epsilon,
                                 model='Ens2D_ResSEN',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)

        del batch_x

        time_taken = time.time() - start_time
        tqdm.write(
            f'Time taken: {time_taken:.3f} | stopped at iter: {i} | eff. SEN: {effectiveness_percentage_sen:.2f}% | eff. Res: {effectiveness_percentage_res:.2f}%')
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config files
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path_res = os.path.join(script_dir, '../config/residualnet_train_config.yaml')
    config_res = read_yaml(config_path_res)
    config_path_sen = os.path.join(script_dir, '../config/SENet.yaml')
    config_sen = read_yaml(config_path_sen)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'Ens2D_ResSEN'  # 'FGSM' or 'BIM'
    epsilon = 4.0
    dataset = 'whole'  # '3s' or 'whole'
    model_version = 'v0'  # or 'old'
    type_of_spec = 'pow'  # 'pow' or 'mag'
    q_res = 10
    q_sen = 80
    '''
    #######################################
    '''

    # load the dataset to work on
    if dataset == 'whole':
        # load the entire ASVSpoof2019 eval dataset
        df_eval = pd.read_csv(os.path.join(script_dir, '..', config_res['df_eval_path']))
    elif dataset == '3s':
        # load the reduced dataset containing only audio >3s
        df_eval = pd.read_csv(os.path.join(script_dir, '..', config_res['df_eval_path_3s']))
    else:
        sys.exit(f'You need to define the dataset to work on, {dataset} is not valid')

    # load the models
    model_res = SpectrogramModel().to(device)
    model_sen = se_resnet34_custom(num_classes=2).to(device)

    model_res.load_state_dict(torch.load(os.path.join(script_dir, '..', config_res['model_path_spec_pow_v0']), map_location=device))
    model_sen.load_state_dict(torch.load(os.path.join(script_dir, '..', config_sen['model_path_spec_pow_v0']), map_location=device))

    model_res.eval()
    model_sen.eval()

    Ens2D_ResSEN(model_res, model_sen, df_eval, config_res, q_res, q_sen, epsilon, device, model_version, dataset)