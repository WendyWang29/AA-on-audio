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
import sys
import librosa
import librosa.feature
import soundfile as sf
from tqdm import tqdm
from src.resnet_model import SpectrogramModel
from src.resnet_features import compute_spectrum
from src.SENet.SENet_model import se_resnet34_custom
from attacks.sp_utils import recover_mag_spec
from check_attacks_utils import get_model_prediction, compute_confidence, get_GT_label
from ResNet_attacks import prepare_dataloader

def plot_2d_grad(grad, model):
    grad = grad.clone().detach().squeeze(dim=0).cpu().numpy()
    plt.figure()
    librosa.display.specshow(grad)
    plt.xlabel('time frames')
    plt.xticks(np.linspace(0, grad.shape[1], 5))
    plt.ylabel('gradient magnitude')
    plt.yticks(np.linspace(0, grad.shape[0], 10))
    plt.colorbar()
    plt.title(f'Gradient: {model}')
    plt.show()


def get_grad(model, model_name, L, batch, y):
    batch.requires_grad = True
    if model_name == 'SENet':
        out = model(batch.unsqueeze(dim=1))
    else:
        out = model(batch)
    loss = L(out, y)
    model.zero_grad()
    loss.backward()
    grad = batch.grad.data
    return grad, out

def grad_average(grad1, grad2):
    grad_avg = (grad1 + grad2) / 2
    return grad_avg

def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std, mean, std

def denormalize(tensor, mean, std):
    return tensor * std + mean

def method_2(grad1, grad2):
    norm1, mean1, std1 = normalize(grad1)
    norm2, mean2, std2 = normalize(grad2)

    # create mask to get values where values exceed mean + 30% of std
    mask1 = torch.abs(norm1) > (0.3)
    mask2 = torch.abs(norm2) > (0.3)

    # init fused tensor
    fused = torch.zeros_like(grad1)

    # if both grads exceed the threshold [mask1 & mask2], take the one with larger magnitude and denormalize
    mask_both = mask1 & mask2
    fused[mask_both] = torch.where(
        torch.abs(grad2[mask_both]) > torch.abs(grad1[mask_both]), grad1[mask_both], grad2[mask_both])

    # if only one grad exceeds...
    mask_only1 = mask1 & ~mask2
    mask_only2 = ~mask1 & mask2
    fused[mask_only1] = grad1[mask_only1]
    fused[mask_only2] = grad2[mask_only2]

    fused = torch.where(
        mask_both | mask_only1 | mask_only2,
        denormalize(fused, mean1 if mask_only1.any() else mean2, std1 if mask_only1.any() else std2),
        fused
    )

    return fused

def model_pred_batch(model, model_name, batch):
    if model_name == 'SENet':
        out = model(batch.unsqueeze(dim=1))
    else:
        out = model(batch)
    label = torch.argmax(out, dim=1)
    return out, label

def plot_gaussian_distr_with_perc(tensor, model, q, perc):
    tensor = tensor.clone().detach().cpu()
    perc = perc.clone().detach().cpu()

    # Flatten the tensor
    flattened_data = tensor.flatten()

    # Compute histogram
    count, bins, ignored = plt.hist(flattened_data, bins=100, density=True, alpha=0.6, color='g')

    # Fit a Gaussian distribution to the data
    mu, std = stats.norm.fit(flattened_data)

    # Plot the histogram
    plt.hist(flattened_data, bins=100, density=True, alpha=0.6, color='g', edgecolor='black')

    # Plot the Gaussian fit
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 100)
    # p = stats.norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2)

    # percentile vertical line
    plt.axvline(perc, color='r', linestyle='--', linewidth=2, label=f'{q}th Percentile')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Gaussian Distribution Fit for {model}')
    plt.legend()
    plt.show()

def compute_silence_percentage(y, silence_thresh=0.02, frame_length=2048, hop_length=512):
    # Compute the Root Mean Square (RMS) energy of the audio signal
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize the RMS values between 0 and 1
    rms_normalized = rms / np.max(rms)

    # Detect silence (RMS values below the silence threshold)
    silence_mask = rms_normalized < silence_thresh

    # Calculate the percentage of silent frames
    silence_percentage = np.sum(silence_mask) / len(silence_mask) * 100

    return silence_percentage

def Ensemble_Attack_SingleFile(file_number,
                               epsilon,
                               eval_path,
                               device,
                               ResNet_model,
                               SENet_model,
                               method,
                               config,
                               type_of_spec='mag'):
    GT_label = get_GT_label(file_number, eval_path)

    # load the original audio file
    clean_path = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac'
    file_number = str(file_number)
    clean_path = os.path.join(clean_path, f'LA_E_{file_number}' + '.flac')
    audio, _ = librosa.load(clean_path, sr=None, duration=240, mono=True)
    og_spec = compute_spectrum(audio, type_of_spec=type_of_spec)

    # clean spec --> mini batch --> clean predictions
    resnet_pred, resnet_pred_label, clean_spec = get_model_prediction(eval_model=ResNet_model,
                                                                      pert_audio=audio,
                                                                      device=device,
                                                                      type_of_spec=type_of_spec,
                                                                      flag=0)

    senet_pred, senet_pred_label, clean_spec = get_model_prediction(eval_model=SENet_model,
                                                                    pert_audio=audio,
                                                                    device=device,
                                                                    type_of_spec=type_of_spec,
                                                                    flag=1)
    print(f'---> File number: {file_number}\n',
          f'--> GT label: {GT_label}\n',
          f'--> Predicted label ResNet: {resnet_pred_label} --- Confidence: {compute_confidence(resnet_pred):.2f} % --- {resnet_pred.tolist()}\n',
          f'--> Predicted label SENet: {senet_pred_label} --- Confidence SENet: {compute_confidence(senet_pred):.2f} % --- {senet_pred.tolist()}')

    # creating the (mini) batches
    batch_x = torch.from_numpy(clean_spec).unsqueeze(dim=0).to(device)
    batch_z = batch_x.clone().to(device)
    batch_y = torch.tensor([int(GT_label)]).to(device)

    L = nn.NLLLoss()

    # ResNet grad computation
    grad_res, out_res = get_grad(model=ResNet_model, model_name='ResNet', L=L, batch=batch_x, y=batch_y)

    # SENet grad computation
    grad_sen, out_sen = get_grad(model=SENet_model, model_name='SENet', L=L, batch=batch_z, y=batch_y)

    #plot_2d_grad(grad_res, 'ResNet')
    #plot_2d_grad(grad_sen, 'SENet')


    if method == 1:
        '''
        BASIC AVERAGING
        '''
        new_grad = grad_average(grad_res, grad_sen)
        plot_2d_grad(grad_res, 'Avg gradient')
        avg_pert_batch = batch_x + epsilon * new_grad.sign()

        avg_pred_res, avg_label_res = model_pred_batch(ResNet_model, 'ResNet', avg_pert_batch)
        avg_pred_sen, avg_label_sen = model_pred_batch(SENet_model, 'SENet', avg_pert_batch)
        print(f'\nAVG grad attack\n'
              f'---> File number: {file_number}\n',
              f'--> GT label: {GT_label}\n',
              f'--> Predicted label ResNet: {avg_label_res.item()} --- Confidence: {compute_confidence(avg_pred_res):.2f} % --- {avg_pred_res.tolist()}\n',
              f'--> Predicted label SENet: {avg_label_sen.item()} --- Confidence SENet: {compute_confidence(avg_pred_sen):.2f} % --- {avg_pred_sen.tolist()}\n')
        del avg_pert_batch
    elif method == 2:
        '''
        NORMALIZATION AND PICKING THE MOST VALUABLE VALUE
        '''
        new_grad = method_2(grad_res, grad_sen)
    elif method == 4:
        '''
        NORMAL FGSM ON SENet
        '''
        pert_batch_z = batch_z + epsilon * grad_sen.sign()
        FGSMSEn_pred_res, FGSMSEn_label_res = model_pred_batch(ResNet_model, 'ResNet', pert_batch_z)
        FGSMSEn_pred_sen, FGSMSEn_label_sen = model_pred_batch(SENet_model, 'SENet', pert_batch_z)
        print(f'\nBasic FGSM on SENet attack\n'
              f'---> File number: {file_number}\n',
              f'--> GT label: {GT_label}\n',
              f'--> Predicted label ResNet: {FGSMSEn_label_res.item()} --- Confidence: {compute_confidence(FGSMSEn_pred_res):.2f} % --- {FGSMSEn_pred_res.tolist()}\n',
              f'--> Predicted label SENet: {FGSMSEn_label_sen.item()} --- Confidence SENet: {compute_confidence(FGSMSEn_pred_sen):.2f} % --- {FGSMSEn_pred_sen.tolist()}\n')

        # convert and save audio
        pert_batch = pert_batch_z.squeeze(dim=0).detach().cpu().numpy()
        og_len = og_spec.shape[1]  # og_spec is the original audio's spectrogram
        sliced_spec = pert_batch[:, :og_len]

        if type_of_spec == 'mag':
            # mag spec is ready already
            mag_spec = sliced_spec
        elif type_of_spec == 'pow':
            # we need to recover the mag spec
            mag_spec = recover_mag_spec(sliced_spec)
        else:
            sys.exit('Invalid type_of_spec')

        phase = np.angle(librosa.stft(y=audio, n_fft=2048, hop_length=512, center=False))

        if og_len < 84:
            phase = phase[:, :og_len]
        elif og_len > 84:
            phase = phase[:, :84]

        p_audio = librosa.istft(mag_spec * np.exp(1j * phase), n_fft=2048, hop_length=512)

        epsilon_str = str(epsilon).replace('.', 'dot')

        if type_of_spec == 'mag':
            folder = 'FGSM_mag_SENet'
            subfolder = f'FGSM_mag_SENet_dataset_{epsilon_str}'
            folder_ = os.path.join(folder, subfolder)
            os.makedirs(folder_, exist_ok=True)
            file_path = os.path.join(folder, subfolder, f'FGSM_mag_SENet_LA_E_{file_number}_{epsilon_str}.flac')
        elif type_of_spec == 'pow':
            folder = 'FGSM_SENet'
            subfolder = f'FGSM_SENet_dataset_{epsilon_str}'
            folder_ = os.path.join(folder, subfolder)
            os.makedirs(folder_, exist_ok=True)
            file_path = os.path.join(folder, subfolder, f'FGSM_SENet_LA_E_{file_number}_{epsilon_str}.flac')

        if os.path.exists(file_path):
            os.remove(file_path)
        sr = 16000
        sf.write(file_path, p_audio, sr, format='FLAC')

        del pert_batch_z

    elif method == 3:
        '''
         NORMAL FGSM ON ResNet
        '''
        pert_batch_x = batch_x + epsilon * grad_res.sign()
        FGSMRes_pred_res, FGSMRes_label_res = model_pred_batch(ResNet_model, 'ResNet', pert_batch_x)
        FGSMRes_pred_sen, FGSMRes_label_sen = model_pred_batch(SENet_model, 'SENet', pert_batch_x)
        print(f'\nBasic FGSM on ResNet attack\n'
              f'---> File number: {file_number}\n',
              f'--> GT label: {GT_label}\n',
              f'--> Predicted label ResNet: {FGSMRes_label_res.item()} --- Confidence: {compute_confidence(FGSMRes_pred_res):.2f} % --- {FGSMRes_pred_res.tolist()}\n',
              f'--> Predicted label SENet: {FGSMRes_label_sen.item()} --- Confidence SENet: {compute_confidence(FGSMRes_pred_sen):.2f} % --- {FGSMRes_pred_sen.tolist()}\n')

        # convert and save audio
        pert_batch = pert_batch_x.squeeze(dim=0).detach().cpu().numpy()
        og_len = og_spec.shape[1]  # og_spec is the original audio's spectrogram
        sliced_spec = pert_batch[:, :og_len]

        if type_of_spec == 'mag':
            # mag spec is ready already
            mag_spec = sliced_spec
        elif type_of_spec == 'pow':
            # we need to recover the mag spec
            mag_spec = recover_mag_spec(sliced_spec)
        else:
            sys.exit('Invalid type_of_spec')

        phase = np.angle(librosa.stft(y=audio, n_fft=2048, hop_length=512, center=False))

        if og_len < 84:
            phase = phase[:, :og_len]
        elif og_len > 84:
            phase = phase[:, :84]

        p_audio = librosa.istft(mag_spec * np.exp(1j * phase), n_fft=2048, hop_length=512)

        epsilon_str = str(epsilon).replace('.', 'dot')

        if type_of_spec == 'mag':
            folder = 'FGSM_mag_ResNet'
            subfolder = f'FGSM_mag_ResNet_dataset_{epsilon_str}'
            folder_ = os.path.join(folder, subfolder)
            os.makedirs(folder_, exist_ok=True)
            file_path = os.path.join(folder, subfolder, f'FGSM_mag_ResNet_LA_E_{file_number}_{epsilon_str}.flac')
        elif type_of_spec == 'pow':
            folder = 'FGSM_ResNet'
            subfolder = f'FGSM_ResNet_dataset_{epsilon_str}'
            folder_ = os.path.join(folder, subfolder)
            os.makedirs(folder_, exist_ok=True)
            file_path = os.path.join(folder, subfolder, f'FGSM_ResNet_LA_E_{file_number}_{epsilon_str}.flac')

        if os.path.exists(file_path):
            os.remove(file_path)
        sr = 16000
        sf.write(file_path, p_audio, sr, format='FLAC')

        del pert_batch_x

    elif method == 5:
        '''
        FINDING THE IMPORTANT VALUES USING 90 PERCENTILE
        '''

        # compute the absolute values
        abs_grad_res = torch.abs(grad_res)
        abs_grad_sen = torch.abs(grad_sen)

        # determine the thresholds
        q_res = 70
        q_sen = 70
        thresh_res = torch.quantile(abs_grad_res, q_res / 100)
        thresh_sen = torch.quantile(abs_grad_sen, q_sen / 100)

        # plot the distribution and percentile
        #(tensor=abs_grad_res, model='ResNet', q=q_res, perc=thresh_res)
        #plot_gaussian_distr_with_perc(tensor=abs_grad_sen, model='SENet', q=q_sen, perc=thresh_sen)

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
        #plot_2d_grad(grad, 'Percentile gradient')

        pert_batch = batch_x + epsilon * grad.sign()

        p_pred_res, p_label_res = model_pred_batch(ResNet_model, 'ResNet', pert_batch)
        p_pred_sen, p_label_sen = model_pred_batch(SENet_model, 'SENet', pert_batch)
        print(f'\nPercentile grad attack {q_res} {q_sen}\n'
              f'---> File number: {file_number}\n',
              f'--> GT label: {GT_label}\n',
              f'--> Predicted label ResNet: {p_label_res.item()} --- Confidence: {compute_confidence(p_pred_res):.2f} % --- {p_pred_res.tolist()}\n',
              f'--> Predicted label SENet: {p_label_sen.item()} --- Confidence SENet: {compute_confidence(p_pred_sen):.2f} % --- {p_pred_sen.tolist()}\n')

        # convert and save audio
        pert_batch = pert_batch.squeeze(dim=0).detach().cpu().numpy()
        og_len = og_spec.shape[1]  # og_spec is the original audio's spectrogram
        sliced_spec = pert_batch[:, :og_len]
        mag_spec = recover_mag_spec(sliced_spec)
        phase = np.angle(librosa.stft(y=audio, n_fft=2048, hop_length=512, center=False))

        if og_len < 84:
            phase = phase[:, :og_len]
        elif og_len > 84:
            phase = phase[:, :84]

        p_audio = librosa.istft(mag_spec * np.exp(1j * phase), n_fft=2048, hop_length=512)

        # silence %
        silence_percentage = compute_silence_percentage(p_audio)
        print(f'Silence: {silence_percentage:.2f}%')

        epsilon_str = str(epsilon).replace('.', 'dot')
        folder = 'Ensemble'
        subfolder = f'QUANT_ENS_{q_res}_{q_sen}_{epsilon_str}'
        folder_ = os.path.join(folder, subfolder)
        os.makedirs(folder_, exist_ok=True)
        file_path = os.path.join(folder, subfolder, f'{subfolder}_{file_number}.flac')

        if os.path.exists(file_path):
            os.remove(file_path)
        sr = 16000
        sf.write(file_path, p_audio, sr, format='FLAC')

        del pert_batch






if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''
    IMPORTANT: set 'mag' or 'pow' models
    '''
    type_of_spec = 'pow'  # 'pow' or 'mag'

    config_ResNet_path = '../config/residualnet_train_config.yaml'
    config_SENet_path = '../config/SENet.yaml'

    config_ResNet = read_yaml(config_ResNet_path)
    config_SENet = read_yaml(config_SENet_path)

    ResNet_model = SpectrogramModel().to(device)
    SENet_model = se_resnet34_custom(num_classes=2).to(device)

    if type_of_spec == 'mag':
        ResNet_model.load_state_dict(
            torch.load(os.path.join('..', config_ResNet['model_path_spec_mag']), map_location=device), strict=False)
        SENet_model.load_state_dict(
            torch.load(os.path.join('..', config_SENet['model_path_spec_mag']), map_location=device), strict=False)
    elif type_of_spec == 'pow':
        ResNet_model.load_state_dict(
            torch.load(os.path.join('..', config_ResNet['model_path_spec_pow']), map_location=device), strict=False)
        SENet_model.load_state_dict(
            torch.load(os.path.join('..', config_SENet['model_path_spec_pow']), map_location=device), strict=False)

    ResNet_model.eval()
    SENet_model.eval()

    print(f'Models ({type_of_spec}) loaded...\n')

    eval_path = os.path.join('..', config_SENet['df_eval_path'])  # eval dataset of ASVSpoof2019

    epsilon = 3.0
    file_number = 1248929
    method = 5

    print(f'Epsilon: {epsilon}, method: {method}\n')
    '''
    method = 1 --> avg of the grads
    method = 2 --> picking
    method = 3 --> normal FGSM on ResNet
    method = 4 --> normal FGSM on SENet
    method = 5 --> percentiles
    '''

    Ensemble_Attack_SingleFile(file_number,
                               epsilon,
                               eval_path,
                               device,
                               ResNet_model,
                               SENet_model,
                               method,
                               config_ResNet,
                               type_of_spec=type_of_spec)
