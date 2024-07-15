import librosa.display
import torch
import os
import matplotlib.pyplot as plt
from torch import Tensor
from src.utils import *
from attacks_utils import load_spec_model, clip_by_tensor, plot_specs_SSA, plot_image
from src.rawnet2_model import RawNet
from torch.utils.data import DataLoader
from src.resnet_utils import LoadAttackData_ResNet, get_features
from src.rawnet_utils import LoadAttackData_RawNet
from attacks_utils import FGSM_perturb_batch_ResNet
from src.rawnet_utils import get_waveform, create_mini_batch_RawNet
from attacks_utils import save_perturbed_audio, get_mini_batch, plot_FFT, FFT_perturb, bpf_att_filter, apply_vad, adjust_grad
from dct import *
from sp_utils import spectrogram_inversion, recover_mag_spec, spectrogram_inversion_batch
from tqdm import tqdm

from src.resnet_features import compute_spectrum
import logging

logging.getLogger('matplotlib.font_manager').disabled = True


'''
########################################
################ RESNET ################
########################################
'''

class ResNetAttack:
    def __init__(self, device, mode='dataset'):
        self.device = device
        config_path = '../config/residualnet_train_config.yaml'
        self.config = read_yaml(config_path)
        self.mode = mode
        model = load_spec_model(device=device, config=self.config).eval()
        self.model = model

        self.eval = pd.read_csv(os.path.join('..', self.config["df_eval_path"]))

        # obtain the data loaders
        if mode == 'dataset':
            eval_labels = dict(zip(self.eval['path'], self.eval['label']))
            file_eval = list(self.eval['path'])

            feat_set = LoadAttackData_ResNet(list_IDs=file_eval,
                                             labels=eval_labels,
                                             win_len=self.config['win_len'],
                                             config=self.config)
            feat_loader = DataLoader(feat_set,
                                     batch_size=self.config['eval_batch_size'],
                                     shuffle=False,
                                     num_workers=15)
            del feat_set, eval_labels
            self.dataset_loader = feat_loader

        if mode != 'dataset' and mode != 'single':
            print('Mode must be "dataset" or "single"')



    def DeepFool_ResNet(self, overshoot=0.00001, max_iter=50, index=0):
        audio_folder = (f'DeepFool_ResNet_dataset')
        current_dir = os.path.dirname(os.path.realpath(__file__))
        audio_folder = os.path.join(current_dir, 'DeepFool_ResNet', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'\nSaving the perturbed audio in {audio_folder}')
        print('\nDeepFool attack on RawNet starts...')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'\nAttacking single file {file} with label {label}')

            audio = get_waveform(wav_path=file, config=self.config)
            spec = compute_spectrum(audio)

            spec_length = spec.shape[1]
            net_input_shape = 28 * 3

            if spec_length < net_input_shape:
                num_repeats = int(net_input_shape / spec_length) + 1
                spec = np.tile(spec, (1, num_repeats))
            spec = spec[:, :net_input_shape]
            spec = Tensor(spec)

            # create the batch
            batch = spec.unsqueeze(dim=0)
            batch = batch.clone().to(self.device)
            batch.requires_grad = True

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            # initial prediction
            output = self.model(batch)
            _, label = torch.max(output, 1)
            label = label.item()

            r_tot = torch.zeros(batch.shape).to(self.device)

            loop_i = 0
            k_i = label
            max_pert = 0.00005

            while k_i == label and loop_i < max_iter:
                print(f'Iteration {loop_i}...')
                # consider only the output relative to the originally predicted class
                output[0, label].backward(retain_graph=True)
                grad_orig = batch.grad.data.clone()

                other_class = 1 - label
                batch.grad.zero_()
                output[0, other_class].backward(retain_graph=True)
                grad_other = batch.grad.data.clone()

                # compute the perturbation
                w = grad_other - grad_orig
                f = output[0, other_class] - output[0, label]
                pert = torch.abs(f) / torch.norm(w.flatten())
                r_i = pert * w /torch.norm(w)
                r_i = torch.clamp(r_i, -max_pert, max_pert)

                # accumulate the total perturbation
                r_tot = r_tot + r_i

                # apply the perturbation
                pert_batch = batch + (1+overshoot) * r_tot
                batch = torch.clamp(pert_batch, -1, 1)
                batch = batch.clone().detach().requires_grad_(True)

                # recompute the model output
                output = self.model(batch)
                k_i = torch.argmax(output.data, 1).item()

                loop_i += 1


    def BIMCut_ResNet(self, epsilon, index=0, iters=100, alpha=0.1):
        # variation of BIM attack in which gradient gets cut like the specs

        epsilon_str = str(epsilon).replace('.','dot')
        audio_folder = f'BIMCut_ResNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'BIMCut_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'Saving the perturbed audio in {audio_folder}')
        print('\nBIM Cut attack starts on ResNet...\n')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'Attacking single file {file} with label {label}')

            # get audio --> spectrogram --> batch for Resnet #############
            audio = get_waveform(wav_path=file, config=self.config)
            spec = compute_spectrum(audio)
            c_spec = spec  # ndarray for plots (spec of original audio)


            ##################
            librosa.display.specshow(c_spec, x_axis='time', sr=16000, y_axis='linear')
            plt.title(f'Power spectrogram for\n {file}', fontsize=8)
            plt.colorbar(format='%+2.0f dB')
            plt.show()
            #################

            spec_length = spec.shape[1]
            net_input_shape = 28 * 3

            if spec_length < net_input_shape:
                num_repeats = int(net_input_shape / spec_length) + 1
                spec = np.tile(spec, (1, num_repeats))
            spec = spec[:, :net_input_shape]
            spec = Tensor(spec)

            # create the batch
            batch = spec.unsqueeze(dim=0)

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            # move to GPU
            batch = batch.to(self.device)
            label = label.to(self.device)
            batch.requires_grad = True

            for i in range(iters):

                if i == iters-1:
                    print('\nCould not perform the attack :(\n')

                self.model.zero_grad()
                out = self.model(batch)

                # early stopping condition
                _, class_pred = torch.max(out, 1)
                if class_pred != label:
                    print(f'Stopped at iteration number {i}')
                    break

                loss = L(out, label)
                loss.backward()
                grad = batch.grad

                # take only 'spec-length' samples from the grad and repeat them
                if spec_length < net_input_shape:
                    temp = grad.clone().cpu().numpy()  # move to cpu
                    t = temp[:, :, :spec_length]  # create the repeating block
                    num_repeats = int(net_input_shape / spec_length) + 1
                    t = np.tile(t, (1, num_repeats))
                    grad = t[:, :, :net_input_shape]

                grad = torch.tensor(grad, device=self.device, requires_grad=True)

                # apply the perturbation and clamp to maintain within the epsilon ball
                p_batch = batch + alpha * grad.sign()
                p_batch_ = torch.clamp(p_batch, min=batch-epsilon, max=batch+epsilon)
                batch.data = p_batch_

            p_spec = batch.squeeze(0).detach().cpu().numpy()
            sliced_spec = p_spec[:, :spec_length]

            ##################
            librosa.display.specshow(p_spec, x_axis='time', sr=16000, y_axis='linear')
            plt.title(f'Power spectrogram for\n perturbed audio with epsilon={epsilon}', fontsize=8)
            plt.colorbar(format='%+2.0f dB')
            plt.show()
            #################

            # spectrogram inversion
            phase = np.angle(librosa.stft(y=audio, n_fft=2048, hop_length=512, center=False))
            phase = phase[:, :net_input_shape]
            mag_spec = recover_mag_spec(sliced_spec)
            p_audio = librosa.istft((mag_spec * np.exp(1j * phase)), n_fft=2048, hop_length=512)

            save_perturbed_audio(file=file_eval[index],
                                 folder=audio_folder,
                                 audio=p_audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='BIMCut')

            ##################
            # librosa.display.specshow(sliced_spec-c_spec, sr=16000, hop_length=512, x_axis='time', y_axis='linear')
            # plt.title('Perturbed spec - clean spec')
            # plt.colorbar()
            # plt.show()
            ##################

            ##################
            audio_name = 'BIMCut_LA_E_8877452_1dot0.flac'
            path = os.path.join('BIMCut_data', f'BIMCut_ResNet_dataset_{epsilon_str}', audio_name)
            audio_n = get_waveform(path, self.config)
            spec = compute_spectrum(audio_n)
            spec_length = spec.shape[1]
            net_input_shape = 28 * 3
            if spec_length < net_input_shape:
                num_repeats = int(net_input_shape / spec_length) + 1
                spec = np.tile(spec, (1, num_repeats))
            spec = spec[:, :net_input_shape]

            diff = p_spec - spec
            librosa.display.specshow(diff, sr=16000, hop_length=512, x_axis='time', y_axis='linear')
            plt.title('Difference between attack-phase spectrogram and \nreconstructed spectrogram to be used as \ninput for ResNet')
            plt.colorbar()
            plt.show()








def FGSM_ResNet(self, epsilon):

        epsilon_str = str(epsilon).replace('.', 'dot')

        if self.mode == 'single':
            pass
        elif self.mode == 'dataset':
            audio_folder = f'FGSM_dataset_{epsilon_str}'
            self.current_dir = os.path.dirname(os.path.abspath(__file__))
            self.audio_folder = os.path.join(self.current_dir, 'FGSM_data', audio_folder)
            os.makedirs(self.audio_folder, exist_ok=True)
            print(f'Saving the perturbed dataset in {self.audio_folder}')

            FGSM_perturb_batch_ResNet(self.dataset_loader, self.model, epsilon, self.config, self.device, self.audio_folder)



    # def SSA_IFGSM_ResNet(self, epsilon, index):
    #     # https://github.com/yuyang-long/SSA/blob/master/attack.py
    #     # Spectrum Simulation Attack
    #
    #     num_iters = 10  # iterations of iterative FGSM
    #     N = 20  # number of spectrum transformations
    #     sigma = 16 # std of random noise
    #     rho = 0.5 # tuning factor
    #     alpha = torch.tensor(epsilon/num_iters)
    #
    #     epsilon_str = str(epsilon).replace('.', 'dot')
    #     audio_folder = f'SSA_ResNet_dataset_{epsilon_str}'
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     audio_folder = os.path.join(current_dir, 'SSA_data', audio_folder)
    #     os.makedirs(audio_folder, exist_ok=True)
    #     print(f'Saving the perturbed audios in {audio_folder}')
    #     print('SSA attack on ReSNet starts...')
    #
    #     if self.mode == 'single':
    #         file_eval = list(self.eval['path'])
    #         label_eval = list(self.eval['label'])
    #
    #         file = file_eval[index]
    #         label = label_eval[index]
    #         label = torch.tensor([label])
    #         print(f'Attacking single file {file} with label {label}')
    #
    #         spec = get_features(wav_path=file,
    #                             features='spec',
    #                             args=self.config,
    #                             X=None,
    #                             cached=True,
    #                             force=False)
    #
    #         #plot_image(spec)
    #         max_val = np.max(spec)
    #         min_val = np.min(spec)
    #
    #         # get 84 time frames long spectrogram
    #         feat_len = spec.shape[1]
    #         net_input_shape = 28 * 3
    #         if feat_len < net_input_shape:
    #             num_repeats = int(net_input_shape / feat_len) + 1
    #             spec = np.tile(spec, (1, num_repeats))
    #         spec = spec[:, :net_input_shape]
    #
    #         # turn the single spec into a mini-batch to be fed to the model
    #         batch = get_mini_batch(spec, self.device)
    #
    #         batch = batch.to(self.device)
    #         label = label.to(self.device)
    #
    #         images_min = clip_by_tensor(batch - epsilon / max_val, min_val, max_val)
    #         images_max = clip_by_tensor(batch + epsilon / max_val, min_val, max_val)
    #
    #         torch.backends.cudnn.enabled = False
    #         L = nn.NLLLoss()
    #
    #         '''
    #         1st for loop is related to I-FGSM
    #         (assuming we work with one spec at a time)
    #         '''
    #         for i in range(num_iters):
    #             noise = 0
    #             '''
    #             2nd loop to get diverse spectrum saliency maps
    #             and then averaging
    #             '''
    #             for n in range(N):
    #                 x = batch.clone()
    #
    #                 # 1 image, (1 channel), shape same as original spec
    #                 gaussian_noise = torch.randn(x.size()[0], spec.shape[0], spec.shape[1]) * (sigma/max_val)
    #                 gaussian_noise = gaussian_noise.to(self.device)
    #
    #                 x_dct = dct_2d(x + gaussian_noise).to(self.device)
    #                 mask = (torch.rand_like(x) * 2 * rho + 1 - rho).to(self.device)
    #                 x_idct = idct_2d(x_dct * mask)  # Hadamard multiplication
    #                 # plot_image(x_idct)
    #                 x_idct = x_idct.requires_grad_(True)
    #
    #                 output_v3 = self.model(x_idct)
    #                 loss = L(output_v3, label)
    #                 loss.backward()
    #                 noise += x_idct.grad.data  # gradient calculation
    #             noise = noise / N   # gradient averaging
    #
    #             x = x + alpha * torch.sign(noise)
    #             x = clip_by_tensor(x, images_min, images_max)
    #             x = x.squeeze(0).detach().cpu().numpy()
    #
    #         # we now have a perturbed spectrogram
    #         sliced_spec = x[:, :feat_len]
    #
    #         plot_specs_SSA(sliced_spec, spec)
    #
    #         audio, _ = spectrogram_inversion(config=self.config,
    #                                         index=index,
    #                                         spec=sliced_spec,
    #                                         phase_info=True)
    #
    #         save_perturbed_audio(file=file_eval[index],
    #                              folder=audio_folder,
    #                              audio=audio,
    #                              sr=16000,
    #                              epsilon=epsilon,
    #                              attack='SSA')
    #
    #     elif self.mode == 'dataset':
    #         torch.backends.cudnn.enabled = False
    #         L = nn.NLLLoss()
    #
    #         df_eval = pd.read_csv(os.path.join('..', self.config["df_eval_path"]))
    #         file_eval = list(df_eval['path'])
    #
    #         for batch_x, batch_y, time_frames, index in tqdm(self.dataset_loader, total=len(self.dataset_loader)):
    #
    #             min_values, _ = torch.min(batch_x, dim=2)
    #             min_values, _ = torch.min(min_values, dim=1)
    #             max_values, _ = torch.max(batch_x, dim=2)
    #             max_values, _ = torch.max(max_values, dim=1)
    #
    #             batch_x = batch_x.to(self.device)
    #             batch_y = batch_y.to(self.device)
    #
    #             temp = epsilon/max_values
    #             sub_values = temp.view(64, 1, 1).to(self.device)
    #
    #             min_values = min_values.view(64, 1, 1).to(self.device)
    #             max_values = max_values.view(64, 1, 1).to(self.device)
    #
    #             images_min = torch.clamp(batch_x - sub_values, min_values, max_values)
    #             images_max = torch.clamp(batch_x + sub_values, min_values, max_values)
    #
    #             for i in range(num_iters):
    #                 noise = 0
    #                 for n in range(N):
    #                     gaussian_noise = torch.randn(batch_x.size()[0], batch_x.size()[1], batch_x.size()[2])
    #                     gaussian_noise = gaussian_noise.to(self.device)
    #                     x_dct = dct_3d(batch_x + gaussian_noise).to(self.device)
    #                     mask = (torch.rand_like(batch_x) * 2 * rho + 1 - rho).to(self.device)
    #                     x_idct = idct_2d(x_dct * mask)
    #                     x_idct = x_idct.requires_grad_(True)
    #
    #                     output_v3 = self.model(x_idct)
    #                     loss = L(output_v3, batch_y)
    #                     loss.backward()
    #                     noise += x_idct.grad.data  # gradient calculation
    #                 noise = noise / N  # gradient averaging
    #
    #                 x = batch_x + alpha * torch.sign(noise)
    #                 x = torch.clamp(x, images_min, images_max)
    #                 x = x.squeeze(0).detach().cpu().numpy()
    #
    #             for i in range(x.shape[0]):
    #                 sliced_spec = x[i][:, :time_frames[i]]
    #                 audio, _ = spectrogram_inversion_batch(config=self.config,
    #                                                        index=index[i],
    #                                                        spec=sliced_spec,
    #                                                        phase_info=True)
    #
    #                 save_perturbed_audio(file=file_eval[index[i]],
    #                                      folder=audio_folder,
    #                                      audio=audio,
    #                                      sr=16000,
    #                                      epsilon=epsilon,
    #                                      attack='SSA')
    #





'''
#########################################
################ RAWNET2 ################
#########################################
'''


class RawNetAttack:
    def __init__(self, device, mode='single'):
        self.device = device
        config_path = '../config/rawnet2.yaml'
        self.config = read_yaml(config_path)
        self.mode = mode

        rawnet_model = RawNet(self.config['model'], device)
        rawnet_model = rawnet_model.to(device)
        rawnet_model.load_state_dict(
            torch.load(os.path.join(self.config['model_path_spec']), map_location=device))
        rawnet_model.eval()
        self.model = rawnet_model

        self.eval = pd.read_csv(os.path.join('..', self.config["df_eval_path"]))

        if mode == 'single' or mode == 'dataset':
            eval_labels = dict(zip(self.eval['path'], self.eval['label']))
            file_eval = list(self.eval['path'])

            feat_set = LoadAttackData_RawNet(list_IDs=file_eval,
                                             labels=eval_labels,
                                             config=self.config)
            feat_loader = DataLoader(feat_set,
                                     batch_size=self.config['eval_batch_size'],
                                     shuffle=False,
                                     num_workers=15)
            del feat_set, eval_labels
            self.dataset_loader = feat_loader

    def DeepFool_RawNet(self, overshoot=1e-5, max_iter=100, index=0):
        audio_folder = f'DeepFool_RawNet_dataset'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'DeepFool_RawNet', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'\nSaving the perturbed audio in {audio_folder}')
        print('\nDeepFool attack on RawNet starts...')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'\nAttacking single file {file} with label {label}')

            audio = get_waveform(wav_path=file, config=self.config)
            audio_len = len(audio)
            network_input_shape = 16000 * 4

            # create the input batch for RawNet2
            if audio_len < network_input_shape:
                num_repeats = int(network_input_shape / audio_len) + 1
                t = np.tile(audio, num_repeats)
                c_audio = t[:network_input_shape]  # clean (tiled) audio
            else:
                c_audio = audio[:network_input_shape]
            batch = create_mini_batch_RawNet(c_audio)

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            batch = batch.clone().to(self.device)
            #label = label.clone().to(self.device) # we're not using the GT label
            batch.requires_grad = True

            # initial prediction
            output = self.model(batch)
            _, label = torch.max(output, 1) # get the predicted label
            label = label.item()

            # tensor for accumulating the perturbation
            r_tot = torch.zeros(batch.shape).to(self.device)

            loop_i = 0
            k_i = label
            max_pert = 1e-5

            while k_i == label and loop_i < max_iter:
                print(f'Iteration {loop_i}...')
                # consider only the output relative to the originally predicted class
                output[0, label].backward(retain_graph=True)
                grad_orig = batch.grad.data.clone()

                other_class = 1 - label
                batch.grad.zero_()
                output[0, other_class].backward(retain_graph=True)
                grad_other = batch.grad.data.clone()

                # compute the perturbation
                w = grad_other - grad_orig
                f = output[0, other_class] - output[0, label]
                pert = torch.abs(f) / torch.norm(w.flatten())
                r_i = pert * w /torch.norm(w)
                r_i = torch.clamp(r_i, -max_pert, max_pert)

                # accumulate the total perturbation
                r_tot = r_tot + r_i

                # apply the perturbation
                pert_batch = batch + (1+overshoot) * r_tot
                batch = torch.clamp(pert_batch, -1, 1)
                batch = batch.clone().detach().requires_grad_(True)

                # recompute the model output
                output = self.model(batch)
                k_i = torch.argmax(output.data, 1).item()

                loop_i += 1

            p_audio = batch.squeeze(0).detach().cpu().numpy()
            sliced_audio = p_audio[:audio_len]
            save_perturbed_audio(file=file_eval[index],
                                 folder=audio_folder,
                                 audio=sliced_audio,
                                 sr=16000,
                                 attack='DeepFool_RawNet')
            print(f'\nAudio file saved.')








    def FGSMC_RawNet(self, epsilon, index=0):
        # FGSM modified (FGSM Cut) so that grad is cut to be the same length of the audio
        # in the batch fed to the RawNet2

        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'FGSMC_RawNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'FGSMC_RawNet', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'\nSaving the perturbed audio in {audio_folder}')
        print('\nFGSMC attack on RawNet starts...')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'\nAttacking single file {file} with label {label}')

            audio = get_waveform(wav_path=file, config=self.config)

            network_input_shape = 16000 * 4
            audio_len = len(audio)

            if audio_len < network_input_shape:
                num_repeats = int(network_input_shape / audio_len) + 1
                t = np.tile(audio, num_repeats)
                c_audio = t[:network_input_shape]  # clean (tiled) audio
            else:
                c_audio = audio[:network_input_shape]

            non_silent_regions = apply_vad(c_audio)  # voice activity detection
            mask = np.where(non_silent_regions, 1.5, 0.5)
            mask = torch.tensor(mask).float().unsqueeze(0).to(self.device)
            batch = create_mini_batch_RawNet(c_audio)  # 64000 samples long

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            batch = batch.to(self.device)
            label = label.to(self.device)
            batch.requires_grad = True
            out = self.model(batch)

            # check the model prediction on the clean audio
            if out[0, 0] > out[0, 1]:
                pred_class = 0
            else:
                pred_class = 1

            if pred_class != label:
                print(f'\nModel is already fooled. Will not perform the attack.')
                p_batch = batch
            else:
                loss = L(out, label)
                loss.backward()
                grad = batch.grad

                # cut and tile the grad
                if audio_len < network_input_shape:
                    num_repeats = int(network_input_shape / audio_len) + 1
                    t = np.tile(grad, num_repeats)
                    grad = t[:, :network_input_shape]
                else:
                    pass

                # apply the perturbation
                p_batch = batch + epsilon * grad.sign() * mask
                out_ = self.model(p_batch)
                print(out_)


            p_batch = p_batch.squeeze(0).detach().cpu().numpy()

            sliced_audio = p_batch[:audio_len]
            save_perturbed_audio(file=file_eval[index],
                                 folder=audio_folder,
                                 audio=sliced_audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='FGSMC_RawNet')
            print(f'\nAudio file saved.')


    def BIMc_RawNet(self, epsilon, alpha=0.01, iters=300, index=0):
        # BIM modified (BIM Cut) so that grad is forced to be the same length of the original audio
        # saves the perturbed audio in the original length of the file

        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'BIM_RawNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'BIM_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'\nSaving the perturbed audio in {audio_folder}')
        print('\nBIM attack on RawNet starts...')

        if self.mode == 'single':

            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'\nAttacking single file {file} with label {label}')

            # load the original audio waveform and create the input batch for rawnet
            audio = get_waveform(wav_path=file, config=self.config)
            c_spec = compute_spectrum(audio)

            non_silent_regions = apply_vad(audio)  # voice activity detection

            # tiled original audio
            network_input_shape = 16000 * 4
            audio_len = len(audio)

            if audio_len < network_input_shape:
                num_repeats = int(network_input_shape / audio_len) + 1
                t = np.tile(audio, num_repeats)
                c_audio = t[:network_input_shape]  # clean (tiled) audio
            else:
                c_audio = audio[:network_input_shape]  # clean original audio (cut)


            length = len(audio)
            batch = create_mini_batch_RawNet(audio) # 64000 samples long

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            batch = batch.to(self.device)
            label = label.to(self.device)

            # BIM attack starts here...
            for i in range(iters):

                if i==iters-1:
                    print('\nCould not perform the attack\n')
                batch.requires_grad = True
                out = self.model(batch)

                # early stopping condition
                if out[0,0] > out[0,1]:
                    pred_class = 0
                else:
                    pred_class = 1

                if pred_class != label:
                    print(f'\nStopped at iteration number {i}\n')

                    # plotting the final gradient FFT to check if BPF worked
                    plot_FFT(item=filtered_sign, title='FFT of the final filtered grad.sign()')
                    break

                self.model.zero_grad()
                loss = L(out, label)
                loss.backward()
                grad = batch.grad

                # take only 'length' samples from grad and repeat it
                network_input_shape = 16000 * 4
                temp = grad.cpu().numpy()
                t = temp[:, :length]

                if length < network_input_shape:
                    num_repeats = int(network_input_shape / length) + 1
                    t = np.tile(t, num_repeats)

                grad = t[:, :network_input_shape]
                grad = torch.tensor(grad, device=self.device, requires_grad=True)

                ##################
                # FILTERING OF THE GRAD
                # apply the BPF + attenuation to the tiled grad
                # sample_rate = 16000
                # lower_cutoff_frequency = 200
                # upper_cutoff_frequency = 5000
                # attenuation_factor = 0.01
                # filtered_sign = bpf_att_filter(grad.sign(), lower_cutoff_frequency, upper_cutoff_frequency, sample_rate,
                #                        attenuation_factor)

                ##################
                # NORMAL CREATION OF THE PERTURBATION
                # perturb the audio signal and clamp the perturbation
                perturbed_batch = batch + alpha * grad.sign()
                eta = torch.clamp(perturbed_batch - batch, min=-epsilon, max=epsilon)
                batch = torch.clamp(batch + eta, min=-1.0, max=1.0).detach_()



                ###################
                # PERTURBATION ON THE FFT DOMAIN
                #batch = FFT_perturb(grad, batch, epsilon, alpha)



            # reconstruct the audio
            p_audio = batch.squeeze(0).detach().cpu().numpy()
            sliced_audio = p_audio[:length]
            save_perturbed_audio(file=file_eval[index],
                                 folder=audio_folder,
                                 audio=sliced_audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='BIM_RawNet')
            print(f'Audio file saved.')

            # plot the difference with the clean spectrogram
            p_spec = compute_spectrum(sliced_audio)
            length = p_spec.shape[1]
            c_spec = c_spec[:, :length]
            temp = p_spec - c_spec
            librosa.display.specshow(temp, sr=16000, hop_length=512, x_axis='time', y_axis='linear')
            plt.title('Perturbed spec - clean spec')
            plt.colorbar(format='%+2.0f dB')
            plt.show()

            # plotting the FFt of the difference between p audio and clean audio
            temp = sliced_audio - audio[:network_input_shape]
            plot_FFT(item=temp, title='FFT of perturbed audio - clean audio')




        elif self.mode == 'dataset':
            pass


    def BIM_RawNet(self, epsilon, alpha=0.001, iters=100, index=0):

        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'BIM_RawNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'BIM_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'Saving the perturbed audio in {audio_folder}')
        print('\nBIM attack on RawNet starts...\n')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'Attacking single file {file} with label {label}')

            audio = get_waveform(wav_path=file, config=self.config)
            ########
            # n = len(audio)
            # audio_fft = np.fft.fft(audio)
            # audio_fft = audio_fft[:n // 2]  # Take the positive frequencies
            #
            # # Step 3: Calculate the frequencies
            # frequencies = np.fft.fftfreq(n, d=1 / 16000)
            # frequencies = frequencies[:n // 2]  # Take the positive frequencies
            #
            # # Step 4: Plot the FFT
            # plt.figure(figsize=(12, 6))
            # plt.plot(frequencies, np.abs(audio_fft) / n)
            # plt.title(f'FFT of {file}')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Amplitude')
            # plt.grid()
            # plt.show()
            # ##########

            c_spec = compute_spectrum(audio)
            length = len(audio)
            batch = create_mini_batch_RawNet(audio) # 64000 samples long

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            batch = batch.to(self.device)
            label = label.to(self.device)

            for i in range(iters):
                batch.requires_grad = True
                out = self.model(batch)

                if out[0,0] > out[0,1]:
                    pred_class = 0
                else:
                    pred_class = 1
                if pred_class != label:
                    print(f'Stopped at iteration number {i}')
                    #temp = grad_filtered.squeeze().cpu()

                    ########
                    # n = len(temp)
                    # p_audio_fft = np.fft.fft(temp)
                    # p_audio_fft = p_audio_fft[:n // 2]  # Take the positive frequencies
                    #
                    # # Step 3: Calculate the frequencies
                    # frequencies = np.fft.fftfreq(n, d=1 / 16000)
                    # frequencies = frequencies[:n // 2]  # Take the positive frequencies
                    #
                    # # Step 4: Plot the FFT
                    # plt.figure(figsize=(12, 6))
                    # plt.plot(frequencies, np.abs(p_audio_fft) / n)
                    # plt.title(f'FFT of grad_filtered')
                    # plt.xlabel('Frequency (Hz)')
                    # plt.ylabel('Amplitude')
                    # plt.xlim(0, 4000)
                    # plt.grid()
                    # plt.show()
                    ##########

                    break

                self.model.zero_grad()
                loss = L(out, label)
                loss.backward()
                grad = batch.grad.data

                # apply the LPF + attenuation
                sample_rate = 16000
                cutoff_frequency = 1000
                attenuation_factor = 1e-07
                #grad_filtered = lpf_att_filter(grad, cutoff_frequency, sample_rate, attenuation_factor)

                #perturbed_batch = batch + alpha * grad_filtered.sign()
                perturbed_batch = batch + alpha * grad.sign()
                eta = torch.clamp(perturbed_batch - batch, min=-epsilon, max=epsilon)
                batch = torch.clamp(batch + eta, min=-1.0, max=1.0).detach_()

            p_audio = batch.squeeze(0).detach().cpu().numpy()

            sliced_audio = p_audio[:length]
            #p_spec = compute_spectrum(sliced_audio)
            #p_spec = compute_spectrum(p_audio)



            # temp = p_spec-c_spec
            # librosa.display.specshow(temp, sr=16000, hop_length=512, x_axis='time', y_axis='linear')
            # plt.title('Perturbed spec - clean spec')
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()

            save_perturbed_audio(file=file_eval[index],
                                folder=audio_folder,
                                audio=p_audio,
                                sr=16000,
                                epsilon=epsilon,
                                attack='BIM_RawNet')


            # check
            # audio_name = 'BIM_RawNet_LA_E_2834763_0dot001.flac'
            # path = os.path.join('BIM_data', f'BIM_RawNet_dataset_{epsilon_str}', audio_name)
            # audio = get_waveform(path, self.config)
            # audio_batch = create_mini_batch_RawNet(audio)
            # audio_batch = audio_batch.squeeze(0).numpy()
            # new_spec = compute_spectrum(audio_batch)
            # temp2 = p_spec - new_spec
            #
            # librosa.display.specshow(temp2, sr=16000, hop_length=512, x_axis='time', y_axis='linear')
            # plt.title('Attack phase 4s audio - reconstructed 4s audio')
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()



            print(f'Audio file saved.')




        elif self.mode == 'dataset':
            pass





















