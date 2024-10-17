from src.rawnet_utils import LoadAttackData_RawNet
from src.utils import *
import os
import gc
from src.ResNet1D.resnet1d_model import SpectrogramModel1D
from src.LCNN_model.LCNN1d_model import LCNN1D

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
import sys
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio

def plot_gaussian(grad):
    # input is a tensor
    grad = grad[:5]
    grad = grad.cpu().numpy().flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(grad, bins=100, alpha=0.5, label='grad gaussian', color='blue', density=True)
    plt.title('Gaussian Distribution of grad_res and grad_sen (First 5 Rows)')
    plt.xlabel('Gradient Values')
    plt.ylabel('Density')
    plt.legend()

    plt.show()

def Ens1D_ResLCNN(config_sen,
                 model_res,
                 model_sen,
                 epsilon,
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_res,
                 q_LCNN):

    epsilon_str = str(epsilon).replace('.', 'dot')
    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    audio_folder = f'Ens1D_ResLCNN_{model_version}_{dataset}_{type_of_spec}_{q_res}_{q_LCNN}_{epsilon_str}'
    audio_folder = os.path.join(current_dir,
                                f'Ens1D_ResLCNN_{model_version}_{type_of_spec}', audio_folder)

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

    n_iters = 50  # max number of BIM iterations
    alpha = epsilon/n_iters  # perturbation to add at each iteration
    print(f'Using n_iters={n_iters} and alpha={alpha}\n')

    # win_length = 2048
    # n_fft = 2048
    # hop_length = 512
    # eps = 1e-20
    # window = 'hann'

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The Ensemble attack on ResNet and SENet starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index, min, max in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        min = min.numpy()
        max = max.numpy()

        w_LCNN = 0.5
        w_res = 1.0

        min_val = torch.min(batch_x)
        max_val = torch.max(batch_x)
        normalized_audio = 2 * (batch_x - min_val) / (max_val - min_val) - 1

        # Scale the normalized audio to range [-1 + eps, 1 - eps]
        scaled_audio = normalized_audio * (1 - epsilon)
        batch_x = scaled_audio

        batch_x = batch_x.to(device)
        batch_z = batch_x.clone().to(device)
        batch_y = batch_y.to(device)

        batch_x.requires_grad = True
        batch_z.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iteration', leave=False) as pbar:
            effectiveness_percentage = 0
            for i in range(n_iters):

                out_LCNN = model_LCNN(batch_x)
                loss_LCNN = L(out_LCNN, batch_y)
                model_LCNN.zero_grad()
                loss_LCNN.backward()
                grad_LCNN = batch_x.grad.data

                out_res = model_res(batch_z)
                loss_res = L(out_res, batch_y)
                model_res.zero_grad()
                loss_res.backward()
                grad_res = batch_z.grad.data

                # # Compute the L2 norm of grad_res and grad_LCNN
                # norm_res = torch.norm(grad_res) + 1e-8  # Adding epsilon to avoid division by zero
                # norm_LCNN = torch.norm(grad_LCNN) + 1e-8  # Adding epsilon to avoid division by zero
                #
                # # Normalize gradients
                # grad_res_normalized = grad_res / norm_res
                # grad_LCNN_normalized = grad_LCNN / norm_LCNN

                grad_res_mean = torch.mean(grad_res)
                grad_res_std = torch.std(grad_res)
                grad_LCNN_mean = torch.mean(grad_LCNN)
                grad_LCNN_std = torch.std(grad_LCNN)

                '''
                compute thresholds
                '''
                abs_grad_LCNN = torch.abs(grad_LCNN)
                abs_grad_res = torch.abs(grad_res)
                thresh_LCNN = torch.quantile(abs_grad_LCNN, q_LCNN/100)
                thresh_res = torch.quantile(abs_grad_res, q_res/100)

                '''
                create new grad
                '''
                matrix = torch.full_like(grad_LCNN, float('inf'))

                # Masks for values that exceed the threshold
                mask_LCNN = abs_grad_LCNN > thresh_LCNN
                mask_res = abs_grad_res > thresh_res

                # Overlap where both grad_res and grad_sen exceed their thresholds
                overlap = mask_LCNN & mask_res

                # For overlapping values, compute the mean of grad_res and grad_sen
                matrix[overlap] = (w_res * grad_res[overlap] + w_LCNN * grad_LCNN[overlap]) / 2


                # For non-overlapping values, take grad_sen values where only grad_sen exceeds the threshold
                matrix[mask_LCNN & ~overlap] = w_LCNN * grad_LCNN[mask_LCNN & ~overlap]

                # For non-overlapping values, take grad_res values where only grad_res exceeds the threshold
                matrix[mask_res & ~overlap] = w_res *grad_res[mask_res & ~overlap]

                # replace inf with 0
                matrix[matrix == float('inf')] = 0

                grad = matrix

                pert_batch = batch_x + alpha * grad.sign()
                #pert_batch = torch.clamp(pert_batch, 0, 1) # clamp so it stays between 0 and 1

                out_pert_LCNN = model_LCNN(pert_batch)
                predicted_labels_LCNN = torch.argmax(out_pert_LCNN, dim=1)
                wrong_predictions_LCNN = (predicted_labels_LCNN != batch_y)
                effectiveness_LCNN = wrong_predictions_LCNN.float().mean()
                effectiveness_percentage_LCNN = effectiveness_LCNN * 100

                out_pert_res = model_res(pert_batch)
                predicted_labels_res = torch.argmax(out_pert_res, dim=1)
                wrong_predictions_res = (predicted_labels_res != batch_y)
                effectiveness_res = wrong_predictions_res.float().mean()
                effectiveness_percentage_res = effectiveness_res * 100

                pbar.set_description(f'BIM iter {i + 1}/{n_iters} | eff. LCNN: {effectiveness_percentage_LCNN:.2f}% | eff. Res: {effectiveness_percentage_res:.2f}%')

                batch_x = batch_z = pert_batch.detach().clone()
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

            audio = batch_x[m]
            audio = librosa.util.normalize(audio)

            save_perturbed_audio(file=file_eval[index[m]],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 attack=attack,
                                 epsilon=epsilon,
                                 model='Ens1D_ResLCNN',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)
        del batch_x


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config files
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path_sen = os.path.join(script_dir, '../config/LCNN1d.yaml')
    config_sen = read_yaml(config_path_sen)
    config_path_res = os.path.join(script_dir, '../config/resnet1d.yaml')
    config_res = read_yaml(config_path_res)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'Ens1D_ResLCNN'   #'FGSM' or 'BIM'
    dataset = 'whole'  # '3s' or 'whole'
    epsilon = 0.01
    model_version = 'v0' # or 'old'
    type_of_spec = 'pow'   # 'pow' or 'mag'
    q_res = 5
    q_LCNN = 90
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
    model_LCNN = LCNN1D().to(device)
    model_res = SpectrogramModel1D().to(device)

    model_LCNN.load_state_dict(torch.load(os.path.join(script_dir, '..', config_sen['model_path_spec_pow_v0']), map_location=device))
    model_res.load_state_dict(torch.load(os.path.join(script_dir, '..', config_res['model_path_spec_pow_v0']), map_location=device))

    model_LCNN.eval()
    model_res.eval()

    Ens1D_ResLCNN(config_sen,
                 model_res,
                 model_LCNN,
                 epsilon,
                 model_version,
                 dataset,
                 df_eval,
                 device,
                 q_res,
                 q_LCNN)
