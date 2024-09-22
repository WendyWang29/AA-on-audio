
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.colorbar').setLevel(logging.ERROR)
logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)

from src.utils import *
import os
import gc
import sys

from tqdm import tqdm
from src.resnet_model import SpectrogramModel
from attacks_utils import save_perturbed_spec, save_perturbed_audio
from src.SENet.SENet_model import se_resnet34_custom
from src.resnet_utils import LoadAttackData_ResNet
from torch.utils.data import DataLoader
import librosa


def EnsembleV1_dataset(epsilon, device, ResNet_model, SENet_model, config, dataset, model_version, type_of_spec):
    """
    Ensemble attack on dataset of audio which is longer than 3s
    :param epsilon: float
    :param eval_path: string
    :param device: string
    :param ResNet_model: SpectrogramModel
    :param SENet_model: CustomResNet
    :param config: dict
    :param dataset: string
    :param model_version: string
    :param type_of_spec: string
    :return:
    """
    q_res = 10
    q_sen = 10

    # ENSEMBLE ATTACK ON EVAL DATASET OF AUDIO WHICH IS LONGER THAN 3S

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script

    if dataset == '3s':
        eval_path = os.path.join(script_dir, '..', 'data/df_eval_19_3s.csv')
    elif dataset == 'whole':
        eval_path = os.path.join(script_dir, '..', 'data/df_eval_19.csv')
    else:
        sys.exit(f'{dataset} is not a valid choice of dataset, should be 3s or whole')

    df_eval = pd.read_csv(eval_path)

    epsilon_str = str(epsilon).replace('.', 'dot')

    # the audio folder will contain the spec folder
    audio_folder = f'QUANT_ENS_{model_version}_{q_res}_{q_sen}_{dataset}_{epsilon_str}'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(current_dir, f'Ensemble', audio_folder)
    spec_folder = os.path.join(current_dir, f'Ensemble', audio_folder, 'spec')

    # os.makedirs(audio_folder, exist_ok=True)
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
                                     type_of_spec=type_of_spec)
    data_loader = DataLoader(feat_set,
                             batch_size=config['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set, labels_eval

    L = nn.NLLLoss()

    effect_res_array = []
    effect_sen_array = []

    win_length = 2048
    n_fft = 2048
    hop_length = 512
    window = 'hann'

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The Ensemble attack starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸')

    for batch_x, batch_y, phase, audio_len, index in tqdm(data_loader, total=len(data_loader)):

        phase = phase.numpy()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_z = batch_x.clone().to(device)

        batch_x.requires_grad = True
        batch_z.requires_grad = True

        '''
        compute the gradients of the 2 models
        '''
        out_res = ResNet_model(batch_x)
        loss_res = L(out_res, batch_y)
        ResNet_model.zero_grad()
        loss_res.backward()
        grad_res = batch_x.grad.data

        out_sen = SENet_model(batch_z.unsqueeze(dim=1))
        loss_sen = L(out_sen, batch_y)
        SENet_model.zero_grad()
        loss_sen.backward()
        grad_sen = batch_z.grad.data

        '''
        compute absolute values
        '''
        abs_grad_res = torch.abs(grad_res)
        abs_grad_sen = torch.abs(grad_sen)

        '''
        find the thresholds
        '''
        thresh_res = torch.quantile(abs_grad_res, q_res / 100)
        thresh_sen = torch.quantile(abs_grad_sen, q_sen / 100)

        '''
        create new grad
        '''
        # create new matrix initiated with infinity
        matrix = torch.full_like(grad_res, float('inf'))

        # populate with important values from grad_res
        matrix[abs_grad_res > thresh_res] = grad_res[abs_grad_res > thresh_res]

        # populate with important values from grad_sen, if overlapping value take the min
        mask = abs_grad_sen > thresh_sen
        matrix[mask] = torch.min(matrix[mask], grad_sen[mask])

        # replace inf with 0
        matrix[matrix == float('inf')] = 0

        grad = matrix

        '''
        apply the perturbation
        '''
        pert_batch = batch_x + epsilon * grad.sign()

        del batch_x, batch_z, grad_res, grad_sen, abs_grad_res, abs_grad_sen, grad, matrix

        '''
        compute the effectiveness
        '''
        # effectiveness on ResNet
        p_batch_out_res = ResNet_model(pert_batch)
        pred_labels_res = torch.argmax(p_batch_out_res, dim=1)
        wrong_predictions_res = (pred_labels_res != batch_y)
        effect_res = wrong_predictions_res.float().mean()
        effect_perc_res = effect_res * 100
        effect_res_array.append(effect_perc_res)
        avg_res = sum(effect_res_array) / len(effect_res_array)

        # effectiveness on SENet
        p_batch_out_sen = SENet_model(pert_batch.unsqueeze(dim=1))
        pred_labels_sen = torch.argmax(p_batch_out_sen, dim=1)
        wrong_predictions_sen = (pred_labels_sen != batch_y)
        effect_sen = wrong_predictions_sen.float().mean()
        effect_perc_sen = effect_sen * 100
        effect_sen_array.append(effect_perc_sen)
        avg_sen = sum(effect_sen_array) / len(effect_sen_array)

        '''
        spec --> audio conversion
        '''
        perturbed_batch = pert_batch.squeeze(0).detach().cpu().numpy()
        for i in range(pert_batch.shape[0]):

            # save the spec as a (1025,93) spec for all specs
            spec = perturbed_batch[i]
            save_perturbed_spec(file=file_eval[index[i]],
                                folder=spec_folder,
                                spec=spec,
                                epsilon=epsilon,
                                attack='Ensemble',
                                model='ResSen',
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
                                 attack='Ensemble',
                                 epsilon=epsilon,
                                 model='ResSen',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)


            del sliced_audio, spec

        del pert_batch
        tqdm.write(f'Effectiveness ResNet: {effect_perc_res:.2f}% | avg.effectiveness ResNet: {avg_res:.2f} \n'
                   f'Effectiveness SENet: {effect_perc_sen:.2f}% | avg.effectiveness SENet: {avg_sen:.2f}\n'
                   f'-------')
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script

    '''
    ##############################
    PARAMETERS
    ##############################
    '''
    model_version = 'v0'
    type_of_spec = 'pow'
    dataset = '3s'  # '3s' or 'whole'
    epsilon = 3.0
    '''
    ##############################
    '''

    assert model_version == 'v0', print('Ensemble attack is implemented only for model version v0')

    config_res_path = os.path.join(script_dir, '../config/residualnet_train_config.yaml')
    config_sen_path = os.path.join(script_dir, '../config/SENet.yaml')

    config_ResNet = read_yaml(config_res_path)
    config_SENet = read_yaml(config_sen_path)

    ResNet_model = SpectrogramModel().to(device)
    SENet_model = se_resnet34_custom(num_classes=2).to(device)

    ResNet_model.load_state_dict(
        torch.load(os.path.join(script_dir, '..', config_ResNet['model_path_spec_pow_v0']), map_location=device),
        strict=False)
    SENet_model.load_state_dict(
        torch.load(os.path.join(script_dir, '..', config_SENet['model_path_spec_pow_v0']), map_location=device),
        strict=False)

    ResNet_model.eval()
    SENet_model.eval()

    print(f'Models loaded...\n')
    print(f'Epsilon: {epsilon} | models version: v0 \n')

    EnsembleV1_dataset(epsilon,
                       device,
                       ResNet_model,
                       SENet_model,
                       config_ResNet,
                       dataset,
                       model_version,
                       type_of_spec)
