from src.rawnet_utils import LoadAttackData_RawNet
from src.utils import *
import os
import gc
from src.ResNet1D.resnet1d_model import SpectrogramModel1D
from src.SENet.senet1d_model import se_resnet341d_custom

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import librosa
import time
import sys
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio


def Ensemble1D_RS(dataset,
                  df_eval,
                  model_version,
                  device,
                  type_of_spec,
                  q_res,
                  q_sen):
    torch.backends.cudnn.enabled = False
    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    script_dir_up = os.path.dirname(script_dir)

    config_path_res = os.path.join(script_dir, '../config/resnet1d.yaml')
    config_res = read_yaml(config_path_res)

    config_path_sen = os.path.join(script_dir, '../config/senet1d.yaml')
    config_sen = read_yaml(config_path_sen)

    # load the models
    res_model = SpectrogramModel1D().to(device)
    sen_model = se_resnet341d_custom(num_classes=2).to(device)

    res_model.load_state_dict(
        torch.load(os.path.join(script_dir_up, config_res['model_path_spec_pow_v0']), map_location=device))
    sen_model.load_state_dict(
        torch.load(os.path.join(script_dir_up, config_sen['model_path_spec_pow_v0']), map_location=device),
        strict=False)

    res_model.eval()
    sen_model.eval()

    # save folder for the audio
    audio_folder = f'QUANT_ENS1D_RS_{model_version}_{q_res}_{q_sen}_{dataset}_None'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(current_dir, f'Ensemble1D_RS', audio_folder)

    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_RawNet(list_IDs=file_eval,
                                     labels=labels_eval,
                                     config=config_sen)
    data_loader = DataLoader(feat_set,
                             batch_size=config['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set, labels_eval

    L = nn.NLLLoss()

    n_iters = 200  # max number of BIM iterations
    alpha = 0.001  # perturbation to add at each iteration

    effect_res_array = []
    effect_sen_array = []

    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')
    print('The Ensemble BIM attack starts...\n')
    print('°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸\n')

    for batch_x, batch_y, audio_len, index in tqdm(data_loader, total=len(data_loader)):
        start_time = time.time()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_z = batch_x.clone().to(device)

        batch_x.requires_grad = True
        batch_z.requires_grad = True

        with tqdm(total=n_iters, desc='BIM iterations', leave=False) as pbar:
            for i in range(n_iters):
                '''
                compute the gradients of the 2 models
                '''
                out_res = res_model(batch_x)
                loss_res = L(out_res, batch_y)
                res_model.zero_grad()
                loss_res.backward()
                grad_res = batch_x.grad.data

                out_sen = sen_model(batch_z)
                loss_sen = L(out_sen, batch_y)
                sen_model.zero_grad()
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
                pert_batch = batch_x + alpha * grad.sign()

                # effectiveness on ResNet
                p_batch_out_res = res_model(pert_batch)
                pred_labels_res = torch.argmax(p_batch_out_res, dim=1)
                wrong_predictions_res = (pred_labels_res != batch_y)
                effect_res = wrong_predictions_res.float().mean()
                effect_perc_res = effect_res * 100

                # effectiveness on RawNet
                p_batch_out_sen = sen_model(pert_batch)
                pred_labels_sen = torch.argmax(p_batch_out_sen, dim=1)
                wrong_predictions_sen = (pred_labels_sen != batch_y)
                effect_sen = wrong_predictions_sen.float().mean()
                effect_perc_sen = effect_sen * 100

                # early stopping
                if (effect_perc_res > 70) and (effect_perc_sen > 70):
                    break

                batch_x = pert_batch.detach().clone()
                batch_x.requires_grad = True
                batch_z = pert_batch.detach().clone()
                batch_z.requires_grad = True

                del grad_res, grad_sen, abs_grad_res, abs_grad_sen, grad, matrix

                if i % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                pbar.update(1)

        pbar.n = min(pbar.total, i + 1)  # Update to the final iteration count
        pbar.refresh()
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
                                 attack=f'EnsembleBIM_RS_{q_res}_{q_sen}',
                                 epsilon=epsilon,
                                 model='ResSen',
                                 model_version=model_version,
                                 type_of_spec=type_of_spec)
        del batch_x, batch_z

        time_taken = time.time() - start_time
        tqdm.write(
            f'Time taken: {time_taken:.3f} | stopped at iter: {i} | Res effect: {effect_perc_res:.2f}%, SEN effect: {effect_perc_sen:.2f}%')
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''
    ########## INSERT PARAMETERS ##########
    '''
    attack = 'Ensemble1D_RS'
    dataset = '3s'  # '3s' or 'whole'
    model_version = 'v0'  # or 'old'
    epsilon = None
    type_of_spec = 'pow'  # 'pow' or 'mag'
    q_res = 10
    q_sen = 10
    '''
    #######################################
    '''

    script_dir = os.path.dirname(os.path.realpath(__file__))  # get directory of current script
    config_path = os.path.join(script_dir, '../config/resnet1d.yaml')
    config = read_yaml(config_path)

    # load the dataset to work on
    if dataset == 'whole':
        # load the entire ASVSpoof2019 eval dataset
        df_eval = pd.read_csv(os.path.join(script_dir, '..', config['df_eval_path']))
    elif dataset == '3s':
        # load the reduced dataset containing only audio >3s
        df_eval = pd.read_csv(os.path.join(script_dir, '..', config['df_eval_path_3s']))
    else:
        sys.exit(f'You need to define the dataset to work on, {dataset} is not valid')

    Ensemble1D_RS(dataset,
                  df_eval,
                  model_version,
                  device,
                  type_of_spec,
                  q_res,
                  q_sen)
