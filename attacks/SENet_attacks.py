from src.utils import *
from src.SENet.SENet_model import se_resnet34_custom
from src.resnet_utils import LoadAttackData_ResNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from attacks.sp_utils import spectrogram_inversion_batch
from attacks_utils import save_perturbed_audio


def FGSM_SENet(epsilon, config, model, df_eval, device):

    # create the folder for the perturbed dataset
    epsilon_str = str(epsilon).replace('.', 'dot')
    audio_folder = f'FGSM_SENet_dataset_{epsilon_str}'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(current_dir, 'FGSM_SENet', audio_folder)
    os.makedirs(audio_folder, exist_ok=True)
    print(f'Saving the perturbed audio in {audio_folder}')
    print('\nFGSM attack on SENet starts...\n')

    # data loader
    file_eval = list(df_eval['path'])
    labels_eval = dict(zip(df_eval['path'], df_eval['label']))

    feat_set = LoadAttackData_ResNet(list_IDs=file_eval,
                                    labels=labels_eval,
                                    win_len=config['win_len'],
                                    config=config)
    data_loader = DataLoader(feat_set,
                             batch_size=config['eval_batch_size'],
                             shuffle=False,
                             num_workers=15)
    del feat_set, labels_eval

    L = nn.NLLLoss()

    print('The attack starts...\n')

    # ATTACK
    # attack loader returns [X_win, y, time_frames, index]
    for batch_x, batch_y, time_frames, index in tqdm(data_loader, total=len(data_loader)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad = True
        out = model(batch_x.unsqueeze(dim=1))
        loss = L(out, batch_y)
        model.zero_grad()
        loss.backward()
        grad = batch_x.grad.data
        perturbed_batch = batch_x + epsilon * grad.sign()

        perturbed_batch = perturbed_batch.squeeze(0).detach()
        perturbed_batch = perturbed_batch.cpu()
        perturbed_batch = perturbed_batch.numpy()

        for i in range(perturbed_batch.shape[0]):
            # working on each row of the matrix of perturbed specs
            sliced_spec = perturbed_batch[i][:, :time_frames[i]]

            audio, _ = spectrogram_inversion_batch(config=config,
                                             index=index[i],
                                             spec=sliced_spec,
                                             phase_info=True)

            save_perturbed_audio(file=file_eval[index[i]],
                                 folder=audio_folder,
                                 audio=audio,
                                 sr=16000,
                                 epsilon=epsilon,
                                 attack='FGSM_SENet')





if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = '../config/SENet.yaml'
    config = read_yaml(config_path)

    df_eval = pd.read_csv(os.path.join('..', config['df_eval_path']))

    model = se_resnet34_custom(num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec']), map_location=device), strict=False)
    model.eval()
    print('Model loaded\n')

    epsilon = 2.0

    FGSM_SENet(epsilon, config, model, df_eval, device)