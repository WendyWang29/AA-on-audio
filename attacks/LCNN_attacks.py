from src.utils import *
import os
import torch.nn.functional as F
from src.LCNN_model.LCNN_model import LCNN
from src.resnet_utils import LoadAttackData_ResNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import time
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio

def prepare_dataloader(attack, epsilon, config, df_eval):
    # create the folder for the perturbed dataset
    epsilon_str = str(epsilon).replace('.', 'dot')
    audio_folder = f'{attack}_LCNN_dataset_{epsilon_str}'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(current_dir, f'{attack}_LCNN', audio_folder)
    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}')
    print(f'\n{attack} attack on LCNN starts...\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_ResNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     win_len=config['win_len'],
                                     config=config)
    data_loader = DataLoader(feat_set,
                             batch_size=config['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set, labels_eval
    return data_loader, file_eval, audio_folder


def apply_smoothing(spec, smoothing_window=5):
    window = torch.hann_window(smoothing_window, periodic=False).to(spec.device)

    # Apply window to the first few columns
    spec[:, :smoothing_window] *= window.unsqueeze(0)

    # Apply window to the last few columns
    spec[:, -smoothing_window:] *= window.flip(0).unsqueeze(0)

    return spec

def BIM_CUT_LCNN(epsilon, config, model, df_eval, device):
    attack = 'BIM_CUT'
    data_loader, file_eval, audio_folder = prepare_dataloader(attack, epsilon, config, df_eval)
    L = nn.NLLLoss()
    print('The attack starts...\n')
    alpha = epsilon/5
    n_iter = 30

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
                smoothed_spec = apply_smoothing(spec[:, :original_len])

                if original_len < net_in_shape:
                    # repeat the smoothed spec
                    repeated_spec = smoothed_spec.repeat(1, (net_in_shape//original_len)+1)
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
                                 attack='BIM_CUT_LCNN')

        time_taken = time.time() - start_time
        tqdm.write(f'Time taken: {time_taken} | Stopped at iter: {stop_iter} | effectiveness: {effectiveness*100:.2f}%')



def FGSM_LCNN(epsilon, config, model, df_eval, device):
    attack = 'FGSM'
    data_loader, file_eval, audio_folder = prepare_dataloader(attack, epsilon, config, df_eval)

    L = nn.NLLLoss()

    print('The attack starts...\n')

    # ATTACK
    # attack loader returns [X_win, y, time_frames, index]
    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True
        out = model(batch_x)
        loss = L(out, batch_y)
        model.zero_grad()
        loss.backward()
        grad = batch_x.grad.data
        perturbed_batch = batch_x + epsilon * grad.sign()

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
                                 attack='FGSM_LCNN')

if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = '../config/LCNN.yaml'
    config = read_yaml(config_path)

    df_eval = pd.read_csv(os.path.join('..', config['df_eval_path']))

    model = LCNN().to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec']), map_location=device), strict=False)
    model.eval()
    print('Model loaded\n')

    epsilon = 2.0

    #FGSM_LCNN(epsilon, config, model, df_eval, device)
    BIM_CUT_LCNN(epsilon, config, model, df_eval, device)

