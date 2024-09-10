from src.utils import *
import os
import gc
from src.resnet_model import SpectrogramModel
from src.resnet_utils import LoadAttackData_ResNet
from src.resnet_features import compute_spectrum
from sp_utils import recover_mag_spec, retrieve_single_audio
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio

def prepare_dataloader(attack, epsilon, config, df_eval, type_of_spec):
    # create the folder for the perturbed dataset
    epsilon_str = str(epsilon).replace('.', 'dot')
    #audio_folder = f'{attack}_ResNet_dataset_{epsilon_str}'
    audio_folder = f'{attack}_{epsilon_str}_dataset'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #audio_folder = os.path.join(current_dir, f'{attack}_ResNet', audio_folder)
    audio_folder = os.path.join(current_dir, 'Ensemble', audio_folder)
    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}')
    print(f'\n{attack} attack on ResNet starts...\n')

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
    del feat_set, labels_eval
    return data_loader, file_eval, audio_folder

def BIM_CUT_ResNet(epsilon, config, model, df_eval, device):
    '''
    only grad cut, no smoothing
    '''
    attack = 'BIM_CUT'
    data_loader, file_eval, audio_folder = prepare_dataloader(attack, epsilon, config, df_eval)
    L = nn.NLLLoss()
    print('The attack starts...\n')
    alpha = epsilon/3
    n_iter = 10

    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True
        for i in range(n_iter):
            out = model(batch_x)
            loss = L(out, batch_y)
            model.zero_grad()
            loss.backward()
            grad = batch_x.grad.data

            # repetition of the grad for each spec
            net_in_shape = 84
            new_grad = torch.zeros_like(grad)
            for n in range(grad.shape[0]):
                spec = grad[n]
                original_len = time_frames[n]
                cut_spec = spec[:, :original_len] # we are actually working on the grad...

                if original_len < net_in_shape:
                    # repeat the smoothed spec
                    num_repeats = int(net_in_shape/original_len) + 1
                    repeated_spec = torch.tile(cut_spec, (1, num_repeats))
                    truncated = repeated_spec[:, :net_in_shape]
                    new_grad[n] = truncated
                else:
                    new_grad[n] = spec

            perturbed_batch = batch_x + alpha * new_grad.sign()
            clipped_batch = torch.clamp(perturbed_batch, batch_x - epsilon, batch_x + epsilon)

            # early stopping
            predictions = model(clipped_batch)
            predicted_labels = torch.argmax(predictions, dim=1)
            wrong_predictions = (predicted_labels != batch_y)
            effectiveness = wrong_predictions.float().mean()
            effectiveness_percentage = effectiveness * 100

            if effectiveness_percentage >= 80:
                stop_iter = i
                break

            del grad, new_grad, loss, out, predictions, predicted_labels, wrong_predictions
            torch.cuda.empty_cache()
            gc.collect()

        perturbed_batch = perturbed_batch.squeeze(0).detach()
        perturbed_batch = perturbed_batch.cpu()
        perturbed_batch = perturbed_batch.numpy()

        for m in range(perturbed_batch.shape[0]):
            # working on each row of the matrix of perturbed specs
            sliced_spec = perturbed_batch[m][:, :time_frames[m]]

            audio, _ = spectrogram_inversion_batch(config=config,
                                                   index=index[m],
                                                   spec=sliced_spec,
                                                   phase_info=True)

            save_perturbed_audio(file=file_eval[index[m]],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='BIM_CUT_ResNet')

            ###
            # checks
            ###
            # audio_1 = audio
            # spec_1 = compute_spectrum(audio_1)
            # spec_0 = perturbed_batch[0]
            # original_len = spec_1.shape[1]
            # num_repeats = int(84/original_len)+1
            # repeated_spec = np.tile(spec_1, (1, num_repeats))
            # truncated = repeated_spec[:, :84]
            # spec_1 = truncated
            # temp = spec_0 - spec_1
            # librosa.display.specshow(temp)
            # plt.show()

        # Free up memory by detaching tensors from the graph and deleting them
        del batch_x, batch_y, perturbed_batch

        time_taken = time.time() - start_time
        tqdm.write(f'Time taken: {time_taken} | Stopped at iter: {stop_iter} | effectiveness: {effectiveness*100:.2f}% | alpha total: {alpha*(stop_iter+1)}')
        torch.cuda.empty_cache()
        gc.collect()

def FGSM_UNCUT_ResNet(epsilon, config, model, df_eval, device):
    attack = 'FGSM_UNCUT'
    data_loader, file_eval, audio_folder = prepare_dataloader(attack, epsilon, config, df_eval)

    L = nn.NLLLoss()

    print('The attack starts...\n')

    # ATTACK
    # attack loader returns [X_win, y, time_frames, index]
    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True
        out = model(batch_x)
        loss = L(out, batch_y)
        model.zero_grad()
        loss.backward()
        grad = batch_x.grad.data
        perturbed_batch = batch_x + epsilon * grad.sign()

        # effectiveness of the attack
        perturbed_batch_out = model(perturbed_batch)
        predicted_labels = torch.argmax(perturbed_batch_out, dim=1)
        wrong_predictions = (predicted_labels != batch_y)
        effectiveness = wrong_predictions.float().mean()
        effectiveness_percentage = effectiveness * 100

        perturbed_batch = perturbed_batch.squeeze(0).detach()
        perturbed_batch = perturbed_batch.cpu()
        perturbed_batch = perturbed_batch.numpy()

        # spec --> audio conversion
        for i in range(perturbed_batch.shape[0]):
            # working on each row of the matrix of perturbed specs
            idx = index[i]

            spec = perturbed_batch[i]
            mag_spec = recover_mag_spec(spec)

            og_audio = retrieve_single_audio(config, index=idx)
            phase = np.angle(librosa.stft(y=og_audio, n_fft=2048, hop_length=512, center=False))
            phase_len = phase.shape[1]
            net_in_shape = 84

            if phase_len < net_in_shape:
                num_repeats = int(net_in_shape / phase_len) + 1
                phase = np.tile(phase, (1, num_repeats))
                phase = phase[:, :net_in_shape]
            else:
                phase = phase[:, :net_in_shape]

            audio = librosa.istft(mag_spec * np.exp(1j * phase), n_fft=2048, hop_length=512)


            save_perturbed_audio(file=file_eval[index[i]],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='FGSM_UNCUT_ResNet')

        time_taken = time.time() - start_time
        tqdm.write(f'Time taken: {time_taken} | effectiveness: {effectiveness_percentage:.2f}% ')


def FGSM_ResNet(epsilon, config, model, df_eval, device):
    attack = 'FGSM'
    data_loader, file_eval, audio_folder = prepare_dataloader(attack, epsilon, config, df_eval)

    L = nn.NLLLoss()

    print('The attack starts...\n')
    effect = []  # for storing the unbalanced effectiveness

    # ATTACK
    # attack loader returns [X_win, y, time_frames, index]
    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):

        start_time = time.time()

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

        # effectiveness of the attack
        perturbed_batch_out = model(perturbed_batch)
        predicted_labels = torch.argmax(perturbed_batch_out, dim=1)
        wrong_predictions = (predicted_labels != batch_y)
        effectiveness = wrong_predictions.float().mean()
        effectiveness_percentage = effectiveness * 100

        effect.append(effectiveness_percentage)
        avg_effect = sum(effect) / len(effect)

        # conversion spec --> audio
        perturbed_batch = perturbed_batch.squeeze(0).detach()
        perturbed_batch = perturbed_batch.cpu()
        perturbed_batch = perturbed_batch.numpy()

        for i in range(perturbed_batch.shape[0]):
            # working on each row of the matrix of perturbed specs
            sliced_spec = perturbed_batch[i][:, :time_frames[i]]

            audio, _ = spectrogram_inversion_batch(config=config,
                                                   index=index[i],
                                                   spec=sliced_spec,
                                                   phase_info=True)

            save_perturbed_audio(file=file_eval[index[i]],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='FGSM_ResNet')
        del perturbed_batch

        time_taken = time.time() - start_time
        tqdm.write(f'Time taken: {time_taken:.3f} | effectiveness: {effectiveness_percentage:.2f}% | avg. effectiveness: {avg_effect:.2f}')
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = '../config/residualnet_train_config.yaml'
    config = read_yaml(config_path)

    df_eval = pd.read_csv(os.path.join('..', config['df_eval_path']))

    model = SpectrogramModel().to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec']), map_location=device), strict=False)
    model.eval()
    print('Model loaded\n')

    epsilon = 4.0

    FGSM_ResNet(epsilon, config, model, df_eval, device)