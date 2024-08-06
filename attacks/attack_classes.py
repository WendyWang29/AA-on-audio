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


    def FGSM_ResNet_c(self, epsilon, index):
        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'FGSMc_ResNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'FGSMc_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'Saving the perturbed audio in {audio_folder}')
        print('\nFGSMc attack starts on ResNet...\n')

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
            c_spec = spec

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

            out = self.model(batch)
            loss = L(out, label)
            loss.backward()
            grad = batch.grad

            p_batch = batch + epsilon * grad.sign()
            p_batch = p_batch.squeeze(0).detach().cpu().numpy()

            sliced_spec = p_batch[:, :spec_length]

            ##################
            librosa.display.specshow(p_batch, x_axis='time', sr=16000, y_axis='linear')
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
                                 attack='FGSMc')

            ##################
            audio_name = 'FGSMc_LA_E_2834763_1dot0.flac'
            path = os.path.join('FGSMc_data', f'FGSMc_ResNet_dataset_{epsilon_str}', audio_name)
            audio_n = get_waveform(path, self.config)
            spec = compute_spectrum(audio_n)
            spec_length = spec.shape[1]
            net_input_shape = 28 * 3
            if spec_length < net_input_shape:
                num_repeats = int(net_input_shape / spec_length) + 1
                spec = np.tile(spec, (1, num_repeats))
            spec = spec[:, :net_input_shape]

            diff = p_batch - spec
            librosa.display.specshow(diff, sr=16000, hop_length=512, x_axis='time', y_axis='linear')
            plt.title(
                'Difference between attack-phase spectrogram and \nreconstructed spectrogram to be used as \ninput for ResNet')
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



    def FGSM_RawNet(self, epsilon, index=0):
        # classic FGSM attack on RawNet
        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'FGSM_RawNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'FGSM_RawNet_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'\nSaving the perturbed audio in {audio_folder}')
        print('\nFGSM attack on RawNet starts...')

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

            batch = create_mini_batch_RawNet(c_audio)  # 64000 samples long

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            batch = batch.to(self.device)
            label = label.to(self.device)
            batch.requires_grad = True

            out = self.model(batch)
            loss = L(out, label)
            loss.backward()
            grad = batch.grad



            # temp = grad[:, :audio_len].clone().detach().cpu().numpy()
            # if audio_len < network_input_shape:
            #     num_repeats = int(network_input_shape / audio_len) + 1
            #     t = np.tile(temp, num_repeats)
            #     grad = t[:, :network_input_shape] # tiled grad
            #     grad = torch.tensor(grad).to(self.device)

            p_batch = batch + epsilon * grad.sign()
            pred = self.model(p_batch)
            p_batch = p_batch.squeeze(0).detach().cpu().numpy()

            sliced_audio = p_batch[:audio_len]

            save_perturbed_audio(file=file_eval[index],
                                 folder=audio_folder,
                                 audio=sliced_audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='FGSM')
            print('\ndone')


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
                temp = grad.clone().detach().cpu()
                if audio_len < network_input_shape:
                    num_repeats = int(network_input_shape / audio_len) + 1
                    t = np.tile(temp, num_repeats)
                    grad = t[:, :network_input_shape]
                    grad = torch.tensor(grad).to(self.device)

                # apply the perturbation
                p_batch = batch + epsilon * grad.sign()
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
            batch = torch.tensor(c_audio).to(self.device)
            batch = batch.unsqueeze(0)
            #batch = create_mini_batch_RawNet(audio) # 64000 samples long

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





















