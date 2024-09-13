import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.colorbar').setLevel(logging.ERROR)
logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)

from src.utils import *
import os
import gc
import sys
import librosa
import soundfile as sf
from tqdm import tqdm
from src.resnet_model import SpectrogramModel
from src.resnet_features import compute_spectrum
from src.SENet.SENet_model import se_resnet34_custom
from attacks.sp_utils import recover_mag_spec
from check_attacks_utils import get_model_prediction, compute_confidence, get_GT_label
from ResNet_attacks import prepare_dataloader
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio


def EnsembleV1_dataset(epsilon, eval_path, device, ResNet_model, SENet_model, config, type_of_spec):
    """
    Ensemble attack on dataset of audio which is longer than 3s
    :param epsilon: float
    :param eval_path: string
    :param device: string
    :param ResNet_model: SpectrogramModel
    :param SENet_model: CustomResNet
    :param config: dict
    :param type_of_spec: string
    :return:
    """
    q_res = 10
    q_sen = 10

    # ENSEMBLE ATTACK ON EVAL DATASET OF AUDIO WHICH IS LONGER THAN 3S

    eval_path = '../data/df_eval_19_3s.csv'
    df_eval = pd.read_csv(eval_path)
    print(f'The dataset contains {len(df_eval)} samples\n')

    data_loader, file_eval, audio_folder = prepare_dataloader(attack=f'QUANT_ENS_{q_res}_{q_sen}',
                                                              epsilon=epsilon,
                                                              config=config,
                                                              df_eval=df_eval,
                                                              type_of_spec=type_of_spec)
    L = nn.NLLLoss()
    effect_res_array = []
    effect_sen_array = []

    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):
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
        pert_batch = pert_batch.squeeze(0).detach().cpu().numpy()
        for i in range(pert_batch.shape[0]):
            # working on each row of the matrix of perturbed specs
            sliced_spec = pert_batch[i][:, :time_frames[i]]

            audio, _ = spectrogram_inversion_batch(config=config,
                                                   index=index[i],
                                                   spec=sliced_spec,
                                                   phase_info=True)

            save_perturbed_audio(file=file_eval[index[i]],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack=f'QUANT_ENS_{q_res}_{q_sen}')
            del audio, sliced_spec
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

    config_ResNet_path = '../config/residualnet_train_config.yaml'
    config_SENet_path = '../config/SENet.yaml'

    config_ResNet = read_yaml(config_ResNet_path)
    config_SENet = read_yaml(config_SENet_path)

    ResNet_model = SpectrogramModel().to(device)
    SENet_model = se_resnet34_custom(num_classes=2).to(device)

    ResNet_model.load_state_dict(torch.load(os.path.join('..', config_ResNet['model_path_spec_pow']), map_location=device), strict=False)
    SENet_model.load_state_dict(torch.load(os.path.join('..', config_SENet['model_path_spec_pow']), map_location=device), strict=False)

    ResNet_model.eval()
    SENet_model.eval()

    print(f'Models loaded...\n')

    eval_path = os.path.join('..', config_SENet['df_eval_path'])  # eval dataset of ASVSpoof2019

    epsilon = 3.0
    type_of_spec = 'pow'

    print(f'Epsilon: {epsilon}, \n')

    EnsembleV1_dataset(epsilon,
                        eval_path,
                        device,
                        ResNet_model,
                        SENet_model,
                        config_ResNet,
                        type_of_spec)
