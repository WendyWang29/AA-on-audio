from src.LCNN_model.LCNN1d_model import LCNN1D
from src.rawnet_utils import LoadAttackData_RawNet
from src.utils import *
import os
import gc

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
import sys
from attacks_utils import save_perturbed_audio, save_perturbed_spec



def BIM_LCNN1D(config, epsilon, model, model_version, dataset, df_eval, device):
    epsilon_str = str(epsilon).replace('.', 'dot')
    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    audio_folder = f'BIM_LCNN1D_{model_version}_{dataset}_{type_of_spec}_{epsilon_str}'
    audio_folder = os.path.join(current_dir, f'BIM_LCNN1D_{model_version}_{type_of_spec}', audio_folder)

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
    alpha = epsilon / n_iters  # perturbation to add at each iteration
    print(f'Using n_iters={n_iters} and alpha={alpha}\n')
    # win_length = 2048
    # n_fft = 2048
    # hop_length = 512
    # eps = 1e-20
    # window = 'hann'

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The BIM attack starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        min_val = torch.min(batch_x)
        max_val = torch.max(batch_x)
        normalized_audio = 2 * (batch_x - min_val) / (max_val - min_val) - 1

        # Scale the normalized audio to range [0, 1 - val]
        scaled_audio = normalized_audio * (1 - epsilon)
        batch_x = scaled_audio

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
                # pert_batch = torch.clamp(pert_batch, 0, 1) # clamp so it stays between 0 and 1

                out_pert = model(pert_batch)
                predicted_labels = torch.argmax(out_pert, dim=1)
                wrong_predictions = (predicted_labels != batch_y)
                effectiveness = wrong_predictions.float().mean()
                effectiveness_percentage = effectiveness * 100

                pbar.set_description(f'BIM iter {i + 1}/{n_iters} | Effectiveness: {effectiveness_percentage:.2f}%')

                batch_x = pert_batch.detach().clone()
                batch_x.requires_grad = True

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
                                 model='LCNN1D',
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
    config_path = os.path.join(script_dir, '../config/LCNN1d.yaml')
    config = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'BIM'  # 'FGSM' or 'BIM'
    dataset = 'whole'  # '3s' or 'whole'
    epsilon = 0.005
    model_version = 'v0'  # or 'old'
    type_of_spec = 'pow'  # 'pow' or 'mag'
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

    model = LCNN1D().to(device)

    if model_version == 'v0':
        model.load_state_dict(
            torch.load(os.path.join(script_dir, '..', config['model_path_spec_pow_v0']), map_location=device))
    elif model_version == 'old':
        model.load_state_dict(
            torch.load(os.path.join(script_dir, '..', config['model_path_spec_pow']), map_location=device))
    else:
        print(f'{model_version} is not defined')
        sys.exit()

    model.eval()
    print(f'ResNet model loaded with weights of version {model_version}\n'
          f'{attack} will be performed with epsilon = {epsilon}, on dataset: {dataset}, using {type_of_spec} spectrograms')

    BIM_LCNN1D(config,
               epsilon,
               model,
               model_version,
               dataset,
               df_eval,
               device)
