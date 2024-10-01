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


def BIM_ResNet1D(config, model, model_version, dataset, df_eval, device):
    # create the folder for the perturbed dataset
    epsilon = None
    type_of_spec = 'pow'  # this was inside the model....
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ex. audio folder: 'BIM_ResNet1D_v0_whole_norm_pow'
    audio_folder = f'BIM_ResNet1D_{model_version}_{dataset}_norm_{type_of_spec}'
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

    n_iters = 200  # max number of BIM iterations
    alpha = 0.0002  # perturbation to add at each iteration
    win_length = 2048
    n_fft = 2048
    hop_length = 512
    eps = 1e-20
    window = 'hann'

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The BIM attack starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True

        for i in range(n_iters):

            out = model(batch_x)
            loss = L(out, batch_y)
            model.zero_grad()
            loss.backward()
            grad = batch_x.grad.data

            pert_batch = batch_x + alpha * grad.sign()

            # early stopping
            out_pert = model(pert_batch)
            predicted_labels = torch.argmax(out_pert, dim=1)
            wrong_predictions = (predicted_labels != batch_y)
            effectiveness = wrong_predictions.float().mean()
            effectiveness_percentage = effectiveness * 100

            if effectiveness_percentage >= 70:
                stop_iter = i
                break

            batch_x = pert_batch.detach().clone()
            batch_x.requires_grad = True

            del grad, loss, out, wrong_predictions
            torch.cuda.empty_cache()
            gc.collect()

        batch_x = batch_x.squeeze(0).detach().cpu().numpy()

        """
        Save audio
        """
        for m in range(batch_x.shape[0]):

            audio = batch_x[m]

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
            f'Time taken: {time_taken:.3f} | stopped at iter: {i}')
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, '../config/resnet1d.yaml')
    config = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'BIM'   #'FGSM' or 'BIM'
    dataset = '3s'  # '3s' or 'whole'
    model_version = 'v0' # or 'old'
    epsilon = None
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


    BIM_ResNet1D(config,
               model,
               model_version,
               dataset,
               df_eval,
               device)