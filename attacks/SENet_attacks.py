from src.utils import *
from src.SENet.SENet_model import se_resnet34_custom
from src.resnet_utils import LoadAttackData_ResNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import librosa
import gc
import sys
import os
from sp_utils import recover_mag_spec, retrieve_single_audio
import time
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio, save_perturbed_spec


def FGSM_SENet(epsilon, config, model, model_version, dataset, type_of_spec, df_eval, device):

    # create the folder for the perturbed dataset
    epsilon_str = str(epsilon).replace('.', 'dot')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ex. audio folder: 'FGSM_SENet_v0_3s_pow_3dot0'
    audio_folder = f'FGSM_SENet_{model_version}_{dataset}_{type_of_spec}_{epsilon_str}'
    audio_folder = os.path.join(current_dir, f'FGSM_SENet_{model_version}_{type_of_spec}', audio_folder)
    spec_folder = os.path.join(current_dir, f'FGSM_SENet_{model_version}_{type_of_spec}', audio_folder, 'spec')

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
        out = model(batch_x.unsqueeze(dim=1))
        loss = L(out, batch_y)
        model.zero_grad()
        loss.backward()
        grad = batch_x.grad.data
        perturbed_batch = batch_x + epsilon * grad.sign()

        del batch_x, grad

        # effectiveness of the attack on ResNet
        perturbed_batch_out = model(perturbed_batch.unsqueeze(dim=1))
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
                                model='SENet',
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
                                 model='SENet',
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
    config_path = os.path.join(script_dir, '../config/SENet.yaml')
    config = read_yaml(config_path)

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'FGSM'  # 'FGSM_3s' or 'FGSM'
    epsilon = 3.0
    dataset = '3s'  # '3s' or 'whole'
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

    model = se_resnet34_custom(num_classes=2).to(device)

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

    print(f'SENet model loaded with weights of version {model_version}\n'
          f'Attack will be performed with epsilon = {epsilon}, on dataset: {dataset}, using {type_of_spec} spectrograms')

    FGSM_SENet(epsilon,
                config,
                model,
                model_version,
                dataset,
                type_of_spec,
                df_eval,
                device)




