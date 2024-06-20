import torch
import os
from src.utils import *
from attacks_utils import load_spec_model, clip_by_tensor, plot_specs_SSA, plot_image
from src.rawnet2_model import RawNet
from torch.utils.data import DataLoader
from src.resnet_utils import LoadAttackData_ResNet, get_features
from src.rawnet_utils import LoadAttackData_RawNet
from attacks_utils import FGSM_perturb_batch_ResNet
from src.rawnet_utils import get_waveform, create_mini_batch_RawNet
from attacks_utils import save_perturbed_audio, get_mini_batch
from dct import *
from sp_utils import spectrogram_inversion, spectrogram_inversion_batch
from tqdm import tqdm



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

        if mode != 'dataset' or mode != 'single':
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
            torch.load(os.path.join('..', self.config['model_path_spec']), map_location=device))
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

    def FGSM_RawNet(self, epsilon):
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
            length = len(audio)
            batch = create_mini_batch_RawNet(audio) # 64000 samples long

            torch.backends.cudnn.enabled = False
            L = nn.NLLLoss()

            batch = batch.to(self.device)
            label = label.to(self.device)

            for i in range(iters):
                batch.requires_grad = True
                out = self.model(batch)
                self.model.zero_grad()
                loss = L(out, label)
                loss.backward()
                grad = batch.grad.data

                perturbed_batch = batch + alpha * grad.sign()
                eta = torch.clamp(perturbed_batch - batch, min=-epsilon, max=epsilon)
                batch = torch.clamp(batch + eta, min=-1.0, max=1.0).detach_()

            batch = batch.squeeze(0).detach().cpu().numpy()

            sliced_audio = batch[:length]
            save_perturbed_audio(file=file_eval[index],
                                folder=audio_folder,
                                audio=sliced_audio,
                                sr=16000,
                                epsilon=epsilon,
                                attack='BIM_RawNet')

        elif self.mode == 'dataset':
            pass

















