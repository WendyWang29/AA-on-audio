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





# def BIM_CUT_ResNet(epsilon, config, model, df_eval, device):
#     '''
#     only grad cut, no smoothing
#     '''
#     attack = 'BIM_CUT'
#     data_loader, file_eval, audio_folder = prepare_dataloader(attack, epsilon, config, df_eval)
#     L = nn.NLLLoss()
#     print('The attack starts...\n')
#     alpha = epsilon/3
#     n_iter = 10
#
#     for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):
#         start_time = time.time()
#
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)
#         batch_x.requires_grad = True
#         for i in range(n_iter):
#             out = model(batch_x)
#             loss = L(out, batch_y)
#             model.zero_grad()
#             loss.backward()
#             grad = batch_x.grad.data
#
#             # repetition of the grad for each spec
#             net_in_shape = 84
#             new_grad = torch.zeros_like(grad)
#             for n in range(grad.shape[0]):
#                 spec = grad[n]
#                 original_len = time_frames[n]
#                 cut_spec = spec[:, :original_len] # we are actually working on the grad...
#
#                 if original_len < net_in_shape:
#                     # repeat the smoothed spec
#                     num_repeats = int(net_in_shape/original_len) + 1
#                     repeated_spec = torch.tile(cut_spec, (1, num_repeats))
#                     truncated = repeated_spec[:, :net_in_shape]
#                     new_grad[n] = truncated
#                 else:
#                     new_grad[n] = spec
#
#             perturbed_batch = batch_x + alpha * new_grad.sign()
#             clipped_batch = torch.clamp(perturbed_batch, batch_x - epsilon, batch_x + epsilon)
#
#             # early stopping
#             predictions = model(clipped_batch)
#             predicted_labels = torch.argmax(predictions, dim=1)
#             wrong_predictions = (predicted_labels != batch_y)
#             effectiveness = wrong_predictions.float().mean()
#             effectiveness_percentage = effectiveness * 100
#
#             if effectiveness_percentage >= 80:
#                 stop_iter = i
#                 break
#
#             del grad, new_grad, loss, out, predictions, predicted_labels, wrong_predictions
#             torch.cuda.empty_cache()
#             gc.collect()
#
#         perturbed_batch = perturbed_batch.squeeze(0).detach()
#         perturbed_batch = perturbed_batch.cpu()
#         perturbed_batch = perturbed_batch.numpy()
#
#         for m in range(perturbed_batch.shape[0]):
#             # working on each row of the matrix of perturbed specs
#             sliced_spec = perturbed_batch[m][:, :time_frames[m]]
#
#             audio, _ = spectrogram_inversion_batch(config=config,
#                                                    index=index[m],
#                                                    spec=sliced_spec,
#                                                    phase_info=True)
#
#             save_perturbed_audio(file=file_eval[index[m]],
#                                  folder=audio_folder,
#                                  audio=audio,
#                                  sr=16000,
#                                  epsilon=epsilon,
#                                  attack='BIM_CUT_ResNet')
#
#             ###
#             # checks
#             ###
#             # audio_1 = audio
#             # spec_1 = compute_spectrum(audio_1)
#             # spec_0 = perturbed_batch[0]
#             # original_len = spec_1.shape[1]
#             # num_repeats = int(84/original_len)+1
#             # repeated_spec = np.tile(spec_1, (1, num_repeats))
#             # truncated = repeated_spec[:, :84]
#             # spec_1 = truncated
#             # temp = spec_0 - spec_1
#             # librosa.display.specshow(temp)
#             # plt.show()
#
#         # Free up memory by detaching tensors from the graph and deleting them
#         del batch_x, batch_y, perturbed_batch
#
#         time_taken = time.time() - start_time
#         tqdm.write(f'Time taken: {time_taken} | Stopped at iter: {stop_iter} | effectiveness: {effectiveness*100:.2f}% | alpha total: {alpha*(stop_iter+1)}')
#         torch.cuda.empty_cache()
#         gc.collect()



def FGSM_ResNet(epsilon, config, model, model_version, dataset, type_of_spec, df_eval, device):

    # create the folder for the perturbed dataset
    epsilon_str = str(epsilon).replace('.', 'dot')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ex. audio folder: 'FGSM_ResNet_v0_3s_pow_3dot0'
    audio_folder = f'FGSM_ResNet_{model_version}_{dataset}_{type_of_spec}_{epsilon_str}'
    audio_folder = os.path.join(current_dir, f'FGSM_ResNet_{model_version}_{type_of_spec}', audio_folder)
    spec_folder = os.path.join(current_dir, f'FGSM_ResNet_{model_version}_{type_of_spec}', audio_folder, 'spec')

    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}...\n')

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

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The FGSM attack starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸')

    effect = []  # for storing the unbalanced effectiveness on Resnet

    win_length = 2048
    n_fft = 2048
    hop_length = 512
    window = 'hann'

    # ########## ATTACK ##########
    for batch_x, batch_y, phase, audio_len, index in tqdm(data_loader, total=len(data_loader)):

        start_time = time.time()
        phase = phase.numpy()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True
        out = model(batch_x)
        loss = L(out, batch_y)
        model.zero_grad()
        loss.backward()
        grad = batch_x.grad.data
        perturbed_batch = batch_x + epsilon * grad.sign()

        del batch_x, grad

        # effectiveness of the attack on ResNet
        perturbed_batch_out = model(perturbed_batch)
        predicted_labels = torch.argmax(perturbed_batch_out, dim=1)
        wrong_predictions = (predicted_labels != batch_y)
        effectiveness = wrong_predictions.float().mean()
        effectiveness_percentage = effectiveness * 100

        effect.append(effectiveness_percentage)
        avg_effect = sum(effect) / len(effect)

        # conversion spec --> audio
        perturbed_batch = perturbed_batch.squeeze(0).detach().cpu().numpy()

        for i in range(perturbed_batch.shape[0]):

            # save the spec as a (1025,93) spec for all specs
            spec = perturbed_batch[i]
            save_perturbed_spec(file=file_eval[index[i]],
                                folder=spec_folder,
                                spec=spec,
                                epsilon=epsilon,
                                attack='FGSM',
                                model='ResNet',
                                model_version=model_version,
                                type_of_spec=type_of_spec)

            # spectrogram inversion
            linear = librosa.db_to_power(spec)
            mag = np.sqrt(linear)

            phase_single_audio = phase[i]

            recon_audio = librosa.istft(mag * np.exp(1j * phase_single_audio),
                                        n_fft=n_fft,
                                        window=window,
                                        win_length=win_length,
                                        hop_length=hop_length,
                                        center=True)

            recon_audio = librosa.util.normalize(recon_audio)

            # cut the audio to original audio length
            sliced_audio = recon_audio[:audio_len[i]]

            save_perturbed_audio(file=file_eval[index[i]],
                                 folder=audio_folder,
                                 audio=sliced_audio,
                                 sr=16000,
                                 attack=attack,
                                 epsilon=epsilon,
                                 model='ResNet',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)


        del perturbed_batch

        time_taken = time.time() - start_time
        tqdm.write(f'Time taken: {time_taken:.3f} | effectiveness: {effectiveness_percentage:.2f}% | avg. effectiveness: {avg_effect:.2f}')
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, '../config/residualnet_train_config.yaml')
    config = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'FGSM'   #'FGSM_3s' or 'FGSM'
    epsilon = 0.0
    dataset = '3s'  # '3s' or 'whole'
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


    model = SpectrogramModel().to(device)

    if model_version == 'v0':
        model.load_state_dict(torch.load(os.path.join(script_dir, '..', config['model_path_spec_pow_v0']), map_location=device))
    elif model_version == 'old':
        model.load_state_dict(torch.load(os.path.join(script_dir, '..', config['model_path_spec_pow']), map_location=device))
    else:
        print(f'{model_version} is not defined')
        sys.exit()

    model.eval()
    print(f'ResNet model loaded with weights of version {model_version}\n'
          f'Attack will be performed with epsilon = {epsilon}, on dataset: {dataset}, using {type_of_spec} spectrograms')

    FGSM_ResNet(epsilon,
                config,
                model,
                model_version,
                dataset,
                type_of_spec,
                df_eval,
                device)