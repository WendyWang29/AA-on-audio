import librosa.display
import torch
import os
import matplotlib.pyplot as plt
from src.utils import *
from attacks_utils import load_spec_model, clip_by_tensor, plot_specs_SSA, plot_image
from src.rawnet2_model import RawNet
from torch.utils.data import DataLoader
from src.resnet_utils import LoadAttackData_ResNet, get_features
from src.rawnet_utils import LoadAttackData_RawNet
from attacks_utils import FGSM_perturb_batch_ResNet
from src.rawnet_utils import get_waveform, create_mini_batch_RawNet
from attacks_utils import save_perturbed_audio, get_mini_batch, lpf_att_filter
from dct import *
from sp_utils import spectrogram_inversion, spectrogram_inversion_batch
from tqdm import tqdm
from src.resnet_features import compute_spectrum



'''
ResNet:
- FGSM single: #TODO
- FGSM dataset: YES
- SSA single: WIP

RawNet2:
- BIM single: YES
- BIM dataset: NO
'''



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

            # old implementation
            FGSM_perturb_batch_ResNet(self.dataset_loader, self.model, epsilon, self.config, self.device,
                                      self.audio_folder)


    def SSA_IFGSM_ResNet(self, epsilon, index):
        # https://github.com/yuyang-long/SSA/blob/master/attack.py
        # Spectrum Simulation Attack

        num_iters = 10  # iterations of iterative FGSM
        N = 20  # number of spectrum transformations
        sigma = 16 # std of random noise
        rho = 0.5 # tuning factor
        alpha = torch.tensor(epsilon/num_iters)

        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'SSA_ResNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'SSA_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'Saving the perturbed audios in {audio_folder}')
        print('SSA attack on ReSNet starts...')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'Attacking single file {file} with label {label}')

            spec = get_features(wav_path=file,
                                features='spec',
                                args=self.config,
                                X=None,
                                cached=True,
                                force=False)

            #plot_image(spec)
            max_val = np.max(spec)
            min_val = np.min(spec)

            # get 84 time frames long spectrogram
            feat_len = spec.shape[1]
            net_input_shape = 28 * 3
            if feat_len < net_input_shape:
                num_repeats = int(net_input_shape / feat_len) + 1
                spec = np.tile(spec, (1, num_repeats))
            spec = spec[:, :net_input_shape]

            # turn the single spec into a mini-batch to be fed to the model
            batch = get_mini_batch(spec, self.device)

            batch = batch.to(self.device)
            label = label.to(self.device)

            images_min = clip_by_tensor(batch - epsilon / max_val, min_val, max_val)
            images_max = clip_by_tensor(batch + epsilon / max_val, min_val, max_val)

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            '''
            1st for loop is related to I-FGSM 
            (assuming we work with one spec at a time)
            '''
            for i in range(num_iters):
                noise = 0
                '''
                2nd loop to get diverse spectrum saliency maps
                and then averaging 
                '''
                for n in range(N):
                    x = batch.clone()

                    # 1 image, (1 channel), shape same as original spec
                    gaussian_noise = torch.randn(x.size()[0], spec.shape[0], spec.shape[1]) * (sigma/max_val)
                    gaussian_noise = gaussian_noise.to(self.device)

                    x_dct = dct_2d(x + gaussian_noise).to(self.device)
                    mask = (torch.rand_like(x) * 2 * rho + 1 - rho).to(self.device)
                    x_idct = idct_2d(x_dct * mask)  # Hadamard multiplication
                    # plot_image(x_idct)
                    x_idct = x_idct.requires_grad_(True)

                    output_v3 = self.model(x_idct)
                    loss = L(output_v3, label)
                    loss.backward()
                    noise += x_idct.grad.data  # gradient calculation
                noise = noise / N   # gradient averaging

                x = x + alpha * torch.sign(noise)
                x = clip_by_tensor(x, images_min, images_max)
                x = x.squeeze(0).detach().cpu().numpy()

            # we now have a perturbed spectrogram
            sliced_spec = x[:, :feat_len]

            plot_specs_SSA(sliced_spec, spec)

            audio, _ = spectrogram_inversion(config=self.config,
                                            index=index,
                                            spec=sliced_spec,
                                            phase_info=True)

            save_perturbed_audio(file=file_eval[index],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='SSA')

        elif self.mode == 'dataset':
            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            df_eval = pd.read_csv(os.path.join('..', self.config["df_eval_path"]))
            file_eval = list(df_eval['path'])

            for batch_x, batch_y, time_frames, index in tqdm(self.dataset_loader, total=len(self.dataset_loader)):

                min_values, _ = torch.min(batch_x, dim=2)
                min_values, _ = torch.min(min_values, dim=1)
                max_values, _ = torch.max(batch_x, dim=2)
                max_values, _ = torch.max(max_values, dim=1)

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                temp = epsilon/max_values
                sub_values = temp.view(64, 1, 1).to(self.device)

                min_values = min_values.view(64, 1, 1).to(self.device)
                max_values = max_values.view(64, 1, 1).to(self.device)

                images_min = torch.clamp(batch_x - sub_values, min_values, max_values)
                images_max = torch.clamp(batch_x + sub_values, min_values, max_values)

                for i in range(num_iters):
                    noise = 0
                    for n in range(N):
                        gaussian_noise = torch.randn(batch_x.size()[0], batch_x.size()[1], batch_x.size()[2])
                        gaussian_noise = gaussian_noise.to(self.device)
                        x_dct = dct_3d(batch_x + gaussian_noise).to(self.device)
                        mask = (torch.rand_like(batch_x) * 2 * rho + 1 - rho).to(self.device)
                        x_idct = idct_2d(x_dct * mask)
                        x_idct = x_idct.requires_grad_(True)

                        output_v3 = self.model(x_idct)
                        loss = L(output_v3, batch_y)
                        loss.backward()
                        noise += x_idct.grad.data  # gradient calculation
                    noise = noise / N  # gradient averaging

                    x = batch_x + alpha * torch.sign(noise)
                    x = torch.clamp(x, images_min, images_max)
                    x = x.squeeze(0).detach().cpu().numpy()

                for i in range(x.shape[0]):
                    sliced_spec = x[i][:, :time_frames[i]]
                    audio, _ = spectrogram_inversion_batch(config=self.config,
                                                           index=index[i],
                                                           spec=sliced_spec,
                                                           phase_info=True)

                    save_perturbed_audio(file=file_eval[index[i]],
                                         folder=audio_folder,
                                         audio=audio,
                                         sr=16000,
                                         epsilon=epsilon,
                                         attack='SSA')






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

    def FGSMS_RawNet(self, epsilon, index=0):
        # "FGSMS" = FGSM attack on Spectrograms using RawNet2

        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'FGSMS_RawNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'FGSMS_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'Saving the perturbed audio in {audio_folder}')
        print('FGSM-S on RawNet2 starts...')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'Attacking single file {file} with label {label}')

            audio = get_waveform(wav_path=file, config=self.config)

            # cuts/tiles to 64000 samples long
            batch = create_mini_batch_RawNet(audio)
            audio = batch.numpy().squeeze() #get the cut/tiled audio

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            batch = batch.to(self.device)
            label = label.to(self.device)

            batch.requires_grad = True
            out = self.model(batch)
            loss = L(out, label)
            self.model.zero_grad()

            loss.backward()
            grad = batch.grad.data
            grad = grad * 500

            # compute the spectrogram from the waveform
            spec_grad = compute_spectrum(grad.cpu().detach().numpy())[0]
            spec = compute_spectrum(audio)

            # perturb the spectrogram
            p_spec = spec + epsilon * np.sign(spec_grad)
            p_spec = spec + np.sign(spec_grad)


            # return to the waveform with spectrogram inversion and original phase
            p_audio, _ = spectrogram_inversion(config=self.config,
                                               index=index,
                                               spec=p_spec,
                                               phase_info=True)
            save_perturbed_audio(file=file_eval[index],
                                 folder=audio_folder,
                                 audio=p_audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='FGSMS_RawNet')
            print('ohoh')


        elif self.mode == 'dataset':
            pass


    def PGD_RawNet(self, epsilon, alpha=0.001, iters=100, index=0):
        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'PGD_RawNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'PGD_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'Saving the perturbed audio in {audio_folder}')
        print('PGD attack on RawNet starts...')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'Attacking single file {file} with label {label}')

            audio = get_waveform(wav_path=file, config=self.config)
            c_spec = compute_spectrum(audio)
            length = len(audio)
            batch = create_mini_batch_RawNet(audio)  # 64000 samples long

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            batch = batch.to(self.device)
            label = label.to(self.device)

            # Initialize perturbed_batch with a random perturbation within epsilon
            p_batch = batch + torch.empty_like(batch).uniform_(-epsilon, epsilon)
            p_batch = torch.clamp(p_batch, min=-1.0, max=1.0)

            for i in range(iters):
                p_batch.requires_grad = True
                out = self.model(p_batch)

                if out[0,0] > out[0,1]:
                    pred_class = 0
                else:
                    pred_class = 1
                if pred_class != label:
                    print(f'Stopped at iteration number {i}')
                    break

                loss = L(out, label)
                loss.backward()
                grad = p_batch.grad.data

                # update step
                p_batch = p_batch + alpha * grad.sign()

                # projection into epsilon ball
                eta = torch.clamp(p_batch - batch, min=-epsilon, max=epsilon)
                p_batch = torch.clamp(batch + eta, min=-1.0, max=1.0).detach_()

            p_audio = p_batch.squeeze(0).detach().cpu().numpy()

            sliced_audio = p_audio[:length]
            p_spec = compute_spectrum(sliced_audio)

            temp = p_spec - c_spec
            librosa.display.specshow(temp, sr=16000, hop_length=512)
            plt.show()

            save_perturbed_audio(file=file_eval[index],
                                 folder=audio_folder,
                                 audio=p_audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='PGD_RawNet')
            print(f'Audio file saved.')


    def BIMc_RawNet(self, epsilon, alpha=0.001, iters=200, index=0):

        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'BIM_RawNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'BIM_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'Saving the perturbed audio in {audio_folder}')
        print('BIM attack on RawNet starts...')

        if self.mode == 'single':
            file_eval = list(self.eval['path'])
            label_eval = list(self.eval['label'])

            file = file_eval[index]
            label = label_eval[index]
            label = torch.tensor([label])
            print(f'Attacking single file {file} with label {label}')

            audio = get_waveform(wav_path=file, config=self.config)
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

                    temp = grad.cpu().numpy()
                    plt.figure()
                    plt.plot(temp[0,:])
                    plt.title('Grad waveform')
                    plt.show()
                    break

                self.model.zero_grad()
                loss = L(out, label)
                loss.backward()
                grad = batch.grad.data

                # apply the LPF + attenuation
                # sample_rate = 16000
                # cutoff_frequency = 1000
                # attenuation_factor = 1e-07
                # grad_filtered = lpf_att_filter(grad, cutoff_frequency, sample_rate, attenuation_factor)

                #perturbed_batch = batch + alpha * grad_filtered.sign()

                grad[:, length:] = 0  #gradient can only be applied to tot samples

                perturbed_batch = batch + alpha * grad.sign()
                eta = torch.clamp(perturbed_batch - batch, min=-epsilon, max=epsilon)
                batch = torch.clamp(batch + eta, min=-1.0, max=1.0).detach_()

                ###############
                network_input_shape = 16000 * 4
                batch_t = batch.cpu().numpy()
                if length < network_input_shape:
                    num_repeats = int(network_input_shape / length) + 1
                    t = np.tile(batch_t, num_repeats)
                batch_t = batch_t[: network_input_shape]
                batch = torch.from_numpy(batch_t).to(self.device)
                ##################

            p_audio = batch.squeeze(0).detach().cpu().numpy()
            sliced_audio = p_audio[:length]

            save_perturbed_audio(file=file_eval[index],
                                folder=audio_folder,
                                audio=sliced_audio,
                                sr=16000,
                                epsilon=epsilon,
                                attack='BIM_RawNet')



            print(f'Audio file saved.')

            #check
            p_spec = compute_spectrum(p_audio)
            audio_name = 'BIM_RawNet_LA_E_2834763_0dot001.flac'
            path = os.path.join('BIM_data', f'BIM_RawNet_dataset_{epsilon_str}', audio_name)
            audio = get_waveform(path, self.config)
            audio_batch = create_mini_batch_RawNet(audio)
            audio_batch = audio_batch.squeeze(0).numpy()
            new_spec = compute_spectrum(audio_batch)
            temp2 = p_spec - new_spec

            librosa.display.specshow(temp2, sr=16000, hop_length=512, x_axis='time', y_axis='linear')
            plt.title('Attack phase 4s audio - reconstructed 4s audio')
            plt.colorbar(format='%+2.0f dB')
            plt.show()




        elif self.mode == 'dataset':
            pass

    def BIM_RawNet(self, epsilon, alpha=0.001, iters=100, index=0):

        epsilon_str = str(epsilon).replace('.', 'dot')
        audio_folder = f'BIM_RawNet_dataset_{epsilon_str}'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(current_dir, 'BIM_data', audio_folder)
        os.makedirs(audio_folder, exist_ok=True)
        print(f'Saving the perturbed audio in {audio_folder}')
        print('BIM attack on RawNet starts...')

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
                grad_filtered = lpf_att_filter(grad, cutoff_frequency, sample_rate, attenuation_factor)

                perturbed_batch = batch + alpha * grad_filtered.sign()
                #perturbed_batch = batch + alpha * grad.sign()
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

















