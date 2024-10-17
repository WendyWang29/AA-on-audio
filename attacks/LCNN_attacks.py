from src.utils import *
import os
import librosa
import gc
import sys

from src.LCNN_model.LCNN_model import LCNN
from src.resnet_utils import LoadAttackData_ResNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import time

from attacks_utils import save_perturbed_audio, save_perturbed_spec


def BIM_LCNN(epsilon,
             config,
             model,
             model_version,
             dataset,
             type_of_spec,
             df_eval,
             device):

    # create the folder for the perturbed dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    epsilon_str = str(epsilon).replace('.', 'dot')

    # audio and spec folder
    audio_folder = f'BIM_LCNN2D_{model_version}_{dataset}_{type_of_spec}'
    audio_folder = os.path.join(current_dir, f'BIM_LCNN2D_{model_version}_{type_of_spec}', audio_folder)
    spec_folder = os.path.join(current_dir, f'BIM_LCNN2D_{model_version}_{type_of_spec}', audio_folder, 'spec')

    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(spec_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}\n')

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
    L = nn.NLLLoss()

    n_iters = 10  # max number of BIM iterations
    alpha = epsilon / n_iters  # perturbation to add at each iteration

    win_length = 2048
    n_fft = 2048
    hop_length = 512
    eps = 1e-20
    window = 'hann'

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The BIM 2D attack on LCNN starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, phase, audio_len, index in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        phase = phase.numpy()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iterations', leave=False) as pbar:
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

                pbar.set_description(
                    f'BIM iter {i + 1}/{n_iters} | eff.: {effectiveness_percentage:.2f}% ')

                batch_x = pert_batch.detach().clone()
                batch_x.requires_grad = True

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
                                model='LCNN2D',
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
                                 model='LCNN2D',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)

        del batch_x

        time_taken = time.time() - start_time
        tqdm.write(
            f'Time taken: {time_taken:.3f} | effectiveness percentage: {effectiveness_percentage}')
        gc.collect()




if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, '../config/LCNN.yaml')
    config = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'BIM'  # 'FGSM' or 'BIM'
    epsilon = 3.0
    dataset = 'whole'  # '3s' or 'whole'
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

    model = LCNN().to(device)

    model.load_state_dict(
        torch.load(os.path.join(script_dir, '..', config['model_path_spec_pow_v0']), map_location=device))

    model.eval()
    print(f'LCNN model loaded with weights of version {model_version}\n'
          f'{attack} will be performed with epsilon = {epsilon}, on dataset: {dataset}, using {type_of_spec} spectrograms')

    BIM_LCNN(epsilon,
             config,
             model,
             model_version,
             dataset,
             type_of_spec,
             df_eval,
             device)
