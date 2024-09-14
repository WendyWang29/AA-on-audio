from src.utils import *
import os
import gc
from src.SENet.SENet_model import se_resnet34_custom
from src.resnet_utils import LoadAttackData_ResNet
from src.resnet_features import compute_spectrum
from sp_utils import recover_mag_spec, retrieve_single_audio
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio, save_perturbed_spec

def FGSM_SENet_3s(epsilon, config, model, df_eval, device):
    epsilon_str = str(epsilon).replace('.', 'dot')

    # the audio folder will contain the spec folder
    audio_folder = f'FGSM_SENet_3s_dataset_{epsilon_str}'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(current_dir, f'{attack}_SENet', audio_folder)
    spec_folder = os.path.join(current_dir, f'{attack}_SENet', audio_folder, 'spec')

    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(spec_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}\n')
    print(f'Saving the perturbed spec in {spec_folder}')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_ResNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     win_len=config['win_len'],
                                     config=config,
                                     type_of_spec='pow')
    data_loader = DataLoader(feat_set,
                             batch_size=config['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set, labels_eval

    # ATTACK
    print('The FGSM attack on the 3s dataset starts...')
    effect = []

    L = nn.NLLLoss()

    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):

        start_time = time.time()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True
        out = model(batch_x.unsqueeze(dim=1))
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
        perturbed_batch = perturbed_batch.squeeze(0).detach()
        perturbed_batch = perturbed_batch.cpu()
        perturbed_batch = perturbed_batch.numpy()

        for i in range(perturbed_batch.shape[0]):
            # working on each row of the matrix of perturbed specs
            sliced_spec = perturbed_batch[i][:, :time_frames[i]]

            save_perturbed_spec(file=file_eval[index[i]],
                                folder=spec_folder,
                                spec=sliced_spec,
                                epsilon=epsilon,
                                attack='FGSM_SENet_3s')

            # spectrogram inversion
            audio, _ = spectrogram_inversion_batch(config=config,
                                                   index=index[i],
                                                   spec=sliced_spec,
                                                   phase_info=True)

            save_perturbed_audio(file=file_eval[index[i]],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='FGSM_SENet_3s')




        del perturbed_batch

        time_taken = time.time() - start_time
        tqdm.write(f'Time taken: {time_taken:.3f} | effectiveness: {effectiveness_percentage:.2f}% | avg. effectiveness: {avg_effect:.2f}')
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = 'config/SENet.yaml'
    config = read_yaml(config_path)

    attack = 'FGSM_3s'
    # 3s dataset
    df_eval = pd.read_csv(os.path.join('..', config['df_eval_path_3s']))

    model = se_resnet34_custom(num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec_pow']), map_location=device), strict=False)
    model.eval()
    print('Model loaded\n')

    epsilon = 3.0

    FGSM_SENet_3s(epsilon, config, model, df_eval, device)