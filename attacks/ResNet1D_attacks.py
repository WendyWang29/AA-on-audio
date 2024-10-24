from src.rawnet_utils import LoadAttackData_RawNet
from src.utils import *
import os
import gc
from src.ResNet1D.resnet1d_model import SpectrogramModel1D


from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
import sys
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio, save_perturbed_spec

def BIMR_ResNet1D(config, epsilon, model, model_version, dataset, df_eval, device):

    epsilon_str = str(epsilon).replace('.', 'dot')
    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    audio_folder = f'BIMR_ResNet1D_{model_version}_{dataset}_{type_of_spec}_{epsilon_str}'
    audio_folder = os.path.join(current_dir, f'BIMR_ResNet1D_{model_version}_{type_of_spec}', audio_folder)

    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}...\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_RawNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     config=config)
    data_loader = DataLoader(feat_set,
                             batch_size=config['eval_batch_size'],
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
    print('The BIMR attack starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index, max_abs, mean in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        max_abs = max_abs.numpy()
        mean = mean.numpy()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Get the original min and max of each sample (before perturbation)
        orig_min = batch_x.min(dim=-1, keepdim=True)[0]
        orig_max = batch_x.max(dim=-1, keepdim=True)[0]

        # Scale batch_x to lie within [orig_min + epsilon, orig_max - epsilon]
        batch_x = (batch_x - orig_min) / (orig_max - orig_min)  # Normalize to [0, 1]
        batch_x = batch_x * (orig_max - orig_min - 2 * epsilon) + (orig_min + epsilon)  # Scale to [orig_min + epsilon, orig_max - epsilon]
        batch_x.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iteration', leave=False) as pbar:
            effectiveness_percentage = 0
            for i in range(n_iters):

                out = model(batch_x)
                loss = L(out, batch_y)
                model.zero_grad()
                loss.backward()
                grad = batch_x.grad.data

                pert_batch = batch_x + alpha * grad.sign()

                out_pert = model(pert_batch)
                predicted_labels = torch.argmax(out_pert, dim=1)
                wrong_predictions = (predicted_labels != batch_y)
                effectiveness = wrong_predictions.float().mean()
                effectiveness_percentage = effectiveness * 100

                pbar.set_description(f'BIM iter {i + 1}/{n_iters} | Effectiveness: {effectiveness_percentage:.2f}%')

                batch_x = pert_batch.detach().clone()
                batch_x.requires_grad = True

                pbar.update(1)

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
                                 epsilon=epsilon,
                                 model='ResNet1D',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)
        del batch_x

        time_taken = time.time() - start_time
        tqdm.write(
            f'Time taken: {time_taken:.3f} | stopped at iter. {i} | effectiveness percentage: {effectiveness_percentage:.2f}%')
        gc.collect()


def BIM_ResNet1D(config, epsilon, model, model_version, dataset, df_eval, device):

    epsilon_str = str(epsilon).replace('.', 'dot')
    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    audio_folder = f'BIM_ResNet1D_{model_version}_{dataset}_{type_of_spec}_{epsilon_str}'
    audio_folder = os.path.join(current_dir, f'BIM_ResNet1D_{model_version}_{type_of_spec}', audio_folder)

    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}...\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_RawNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     config=config)
    data_loader = DataLoader(feat_set,
                             batch_size=config['eval_batch_size'],
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
    print('The BIM attack starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index, max_abs, mean in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        max_abs = max_abs.numpy()
        mean = mean.numpy()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iteration', leave=False) as pbar:
            effectiveness_percentage = 0
            for i in range(n_iters):

                out = model(batch_x)
                loss = L(out, batch_y)
                model.zero_grad()
                loss.backward()
                grad = batch_x.grad.data

                pert_batch = batch_x + alpha * grad.sign()

                out_pert = model(pert_batch)
                predicted_labels = torch.argmax(out_pert, dim=1)
                wrong_predictions = (predicted_labels != batch_y)
                effectiveness = wrong_predictions.float().mean()
                effectiveness_percentage = effectiveness * 100

                pbar.set_description(f'BIM iter {i + 1}/{n_iters} | Effectiveness: {effectiveness_percentage:.2f}%')

                batch_x = pert_batch.detach().clone()
                batch_x.requires_grad = True

                pbar.update(1)

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
                                 epsilon=epsilon,
                                 model='ResNet1D',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)
        del batch_x

        time_taken = time.time() - start_time
        tqdm.write(
            f'Time taken: {time_taken:.3f} | stopped at iter. {i} | effectiveness percentage: {effectiveness_percentage:.2f}%')
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, '../config/resnet1d.yaml')
    config = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'BIMR'   #'FGSM' or 'BIM'
    dataset = 'whole'  # '3s' or 'whole'
    epsilon = 0.025
    model_version = 'v0' # or 'old'
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


    model = SpectrogramModel1D().to(device)

    if model_version == 'v0':
        model.load_state_dict(torch.load(os.path.join(script_dir, '..', config['model_path_spec_pow_v0']), map_location=device))
    elif model_version == 'old':
        model.load_state_dict(torch.load(os.path.join(script_dir, '..', config['model_path_spec_pow']), map_location=device))
    else:
        print(f'{model_version} is not defined')
        sys.exit()

    model.eval()
    print(f'ResNet model loaded with weights of version {model_version}\n'
          f'{attack} will be performed with epsilon = {epsilon}, on dataset: {dataset}, using {type_of_spec} spectrograms')

    BIMR_ResNet1D(config,
                 epsilon,
                 model,
                 model_version,
                 dataset,
                 df_eval,
                 device)

    # BIM_ResNet1D(config,
    #              epsilon,
    #              model,
    #              model_version,
    #              dataset,
    #              df_eval,
    #              device)